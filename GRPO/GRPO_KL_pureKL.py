# GRPO_pure_KL.py
import os
import json
import time
import random
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

from baseline_funs import build_prompt, extract_assistant_response, generate_batch
from drgrpo_grader_new import r1_zero_reward_fn
from SFT import (
    tokenize_prompt_and_output,
    truncate_to_answer_end,
)

LEVEL_NAMES = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    lr_min: float,
):
    """
    线性 warmup + cosine decay。
    这里的 total_steps 指 optimizer.step() 的总次数，而不是外层 GRPO step 数。
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    if warmup_steps == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=lr_min,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=lr_min,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
    return scheduler


def build_level_buckets(question_dataset: List[Dict[str, Any]]) -> dict[str, list[dict]]:
    """
    按 Level 分桶。
    若某条数据没有 Level，默认放到 Level 3。
    """
    buckets = {lvl: [] for lvl in LEVEL_NAMES}

    for ex in question_dataset:
        lvl = ex.get("Level", ex.get("level", "Level 3"))
        if lvl not in buckets:
            lvl = "Level 3"
        buckets[lvl].append(ex)

    # 防止某个 level 空桶导致后续采样报错
    non_empty = {k: v for k, v in buckets.items() if len(v) > 0}
    if len(non_empty) == 0:
        raise ValueError("question_dataset is empty after level bucketing.")

    return buckets

def get_response_distribution(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    """
    返回 response 对齐位置上的：
    - 全词表 log_probs
    - sampled token 的 log_probs / probs
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits                       # (B, T, V)
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)

    selected_log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)                                # (B, T)

    selected_probs = selected_log_probs.exp()    # (B, T)

    return {
        "logits": logits,
        "log_probs_all": log_probs_all,
        "selected_log_probs": selected_log_probs,
        "selected_probs": selected_probs,
    }

def forward_kl_old_to_policy(
    old_log_probs_all: torch.Tensor,
    policy_log_probs_all: torch.Tensor,
):
    """
    计算每个 token 位置上的 forward KL:
        KL(old_policy || policy)
    返回形状: (B, T)
    """
    old_probs_all = old_log_probs_all.exp()
    kl = (old_probs_all * (old_log_probs_all - policy_log_probs_all)).sum(dim=-1)
    return kl


def pad_and_stack_2d(tensors: list[torch.Tensor], pad_value: float = 0.0):
    max_len = max(x.size(1) for x in tensors)
    out = []
    for x in tensors:
        if x.size(1) < max_len:
            x = F.pad(x, (0, max_len - x.size(1)), value=pad_value)
        out.append(x)
    return torch.cat(out, dim=0)


def pad_and_stack_3d(tensors: list[torch.Tensor], pad_value: float = 0.0):
    max_len = max(x.size(1) for x in tensors)
    out = []
    for x in tensors:
        if x.size(1) < max_len:
            x = F.pad(x, (0, 0, 0, max_len - x.size(1)), value=pad_value)
        out.append(x)
    return torch.cat(out, dim=0)

def get_curriculum_level_probs(
    local_step: int,
    n_grpo_steps: int,
    start_probs: dict[str, float] | None = None,
    end_probs: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    根据当前 local_step 返回一个 level -> prob 的分布。
    使用线性插值，让难题占比逐步升高，但低难题始终保留。
    """
    if start_probs is None:
        start_probs = {
            "Level 1": 0.35,
            "Level 2": 0.30,
            "Level 3": 0.20,
            "Level 4": 0.10,
            "Level 5": 0.05,
        }

    if end_probs is None:
        end_probs = {
            "Level 1": 0.10,
            "Level 2": 0.15,
            "Level 3": 0.20,
            "Level 4": 0.25,
            "Level 5": 0.30,
        }

    # progress ∈ [0, 1]
    if n_grpo_steps <= 1:
        progress = 1.0
    else:
        progress = (local_step - 1) / (n_grpo_steps - 1)
        progress = max(0.0, min(1.0, progress))

    probs = {}
    for lvl in LEVEL_NAMES:
        s = start_probs.get(lvl, 0.0)
        e = end_probs.get(lvl, 0.0)
        probs[lvl] = (1.0 - progress) * s + progress * e

    # 归一化
    total = sum(probs.values())
    probs = {k: v / total for k, v in probs.items()}
    return probs


def sample_curriculum_batch(
    level_buckets: dict[str, list[dict]],
    batch_size: int,
    level_probs: dict[str, float],
) -> List[Dict[str, Any]]:
    """
    先按 level_probs 采样 level，再从对应桶里采样题目。
    若某个 level 空桶，则自动从非空桶重分配。
    """
    non_empty_levels = [lvl for lvl, items in level_buckets.items() if len(items) > 0]
    if len(non_empty_levels) == 0:
        raise ValueError("All level buckets are empty.")

    # 只保留非空桶概率
    filtered_probs = {lvl: level_probs.get(lvl, 0.0) for lvl in non_empty_levels}
    total = sum(filtered_probs.values())
    if total <= 0:
        # 兜底：均匀分布
        filtered_probs = {lvl: 1.0 / len(non_empty_levels) for lvl in non_empty_levels}
    else:
        filtered_probs = {lvl: p / total for lvl, p in filtered_probs.items()}

    levels = list(filtered_probs.keys())
    probs = [filtered_probs[lvl] for lvl in levels]

    sampled = []
    for _ in range(batch_size):
        chosen_level = random.choices(levels, weights=probs, k=1)[0]
        sampled.append(random.choice(level_buckets[chosen_level]))

    return sampled

### 计算相对奖励
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """
    rollout_responses: list[str], length = rollout_batch_size
    repeated_ground_truths: list[str], length = rollout_batch_size
    """
    raw_reward_list = []
    format_reward_list = []

    for response, gt in zip(rollout_responses, repeated_ground_truths):
        res = reward_fn(response, gt)
        raw_reward_list.append(float(res["reward"]))
        format_reward_list.append(float(res.get("format_reward", 0.0)))

    raw_rewards = torch.tensor(raw_reward_list, dtype=torch.float32)

    batch_size = raw_rewards.shape[0]
    if batch_size % group_size != 0:
        raise ValueError("batch_size must be divisible by group_size")

    num_groups = batch_size // group_size
    reshaped_rewards = raw_rewards.view(num_groups, group_size)

    group_means = reshaped_rewards.mean(dim=1, keepdim=True)
    centered = reshaped_rewards - group_means

    if normalize_by_std:
        group_stds = reshaped_rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages_reshaped = centered / (group_stds + advantage_eps)
    else:
        group_stds = reshaped_rewards.std(dim=1, keepdim=True, unbiased=False)
        advantages_reshaped = centered

    advantages = advantages_reshaped.reshape(-1)

    metadata = {
        "reward_mean": raw_rewards.mean().detach(),
        "reward_std": raw_rewards.std(unbiased=False).detach(),
        "reward_max": raw_rewards.max().detach(),
        "reward_min": raw_rewards.min().detach(),
        "format_reward_mean": torch.tensor(format_reward_list, dtype=torch.float32).mean().detach(),
        "num_rollouts": torch.tensor(float(batch_size)),
        "num_groups": torch.tensor(float(num_groups)),
    }

    return advantages, raw_rewards, metadata

### 目标函数

def compute_grpo_kl_loss(
    advantages: torch.Tensor,                 # (B, 1)
    policy_log_probs_all: torch.Tensor,       # (B, T, V)
    policy_log_probs: torch.Tensor,           # (B, T) sampled token
    old_log_probs: torch.Tensor,              # (B, T) sampled token
    old_log_probs_all: torch.Tensor,          # (B, T, V)
    response_mask: torch.Tensor,              # (B, T)
    beta: float,
    clip_low: float,
    clip_high: float,
):
    """
    纯 KL 版 GRPO:
      clip(r, clip_low, clip_high) * A
      - beta * KL(old_policy || policy)

    不再依赖低概率阈值、proxy 分布、负优势筛选等条件。
    """
    ratios = torch.exp(policy_log_probs - old_log_probs)              # (B, T)
    clipped_ratios = torch.clamp(ratios, clip_low, clip_high)         # 双边界 [0.8, 1.2]

    pg_obj = clipped_ratios * advantages                              # (B, T)

    kl_per_token = forward_kl_old_to_policy(
        old_log_probs_all=old_log_probs_all,
        policy_log_probs_all=policy_log_probs_all,
    )                                                                 # (B, T)

    per_token_obj = pg_obj - beta * kl_per_token
    loss = -per_token_obj

    token_was_clipped = (ratios != clipped_ratios) & response_mask.bool()

    metadata = {
        "clip_fraction": token_was_clipped.float().sum().detach()
                         / response_mask.float().sum().clamp_min(1.0),
        "kl_mean": (kl_per_token * response_mask.float()).sum().detach()
                   / response_mask.float().sum().clamp_min(1.0),
        "sampled_prob_mean": (policy_log_probs.exp() * response_mask.float()).sum().detach()
                             / response_mask.float().sum().clamp_min(1.0),
    }

    return loss, metadata


### 只计算response的loss
def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    mask = mask.to(tensor.dtype)

    if dim is None:
        denom = mask.sum().clamp_min(1.0)
        return (tensor * mask).sum() / denom

    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (tensor * mask).sum(dim=dim) / denom

### 一个microbatch中的GRPO更新
def grpo_microbatch_train_step(
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,

    policy_log_probs: torch.Tensor,
    policy_log_probs_all: torch.Tensor,

    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_log_probs_all: torch.Tensor,

    kl_beta: float,
    clip_low: float,
    clip_high: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on one GRPO microbatch.
    这里只保留纯 KL 正则路径。
    """
    per_token_loss, loss_meta = compute_grpo_kl_loss(
        advantages=advantages,
        policy_log_probs_all=policy_log_probs_all,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        old_log_probs_all=old_log_probs_all,
        response_mask=response_mask,
        beta=kl_beta,
        clip_low=clip_low,
        clip_high=clip_high,
    )

    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    loss = per_example_loss.sum()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = dict(loss_meta)
    return loss, metadata


###一次GRPOstep, 包括gradient_accumulation_steps次grpo_microbatch_train_step
def grpo_step(
    model,
    tokenizer,
    optimizer,
    scheduler,
    prompt_strs: list[str],
    response_strs: list[str],
    advantages: torch.Tensor,
    device: torch.device,
    gradient_accumulation_steps: int,

    old_log_probs: torch.Tensor,
    old_log_probs_all: torch.Tensor,

    kl_beta: float = 0.01,
    clip_low: float = 0.8,
    clip_high: float = 1.2,

    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    assert len(prompt_strs) == len(response_strs)
    batch_size = len(prompt_strs)
    assert batch_size > 0
    assert batch_size % gradient_accumulation_steps == 0, (
        "batch_size must be divisible by gradient_accumulation_steps"
    )

    model.train()
    optimizer.zero_grad()

    microbatch_size = batch_size // gradient_accumulation_steps
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id.")

    total_loss = 0.0
    total_response_tokens = 0

    total_clip_fraction = 0.0
    clip_fraction_count = 0

    total_kl_mean = 0.0
    kl_mean_count = 0

    for micro_idx in range(gradient_accumulation_steps):
        start = micro_idx * microbatch_size
        end = (micro_idx + 1) * microbatch_size

        micro_prompts = prompt_strs[start:end]
        micro_responses = response_strs[start:end]

        batch = tokenize_prompt_and_output(
            prompt_strs=micro_prompts,
            output_strs=micro_responses,
            tokenizer=tokenizer,
        )

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)
        attention_mask = (input_ids != pad_token_id)

        forward_out = get_response_distribution(
            model=model,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        policy_log_probs_all = forward_out["log_probs_all"]
        policy_log_probs = forward_out["selected_log_probs"]
        micro_advantages = advantages[start:end].to(device).unsqueeze(1)

        micro_old_log_probs = old_log_probs[start:end].to(device)
        micro_old_log_probs = micro_old_log_probs[:, :policy_log_probs.size(1)]

        micro_old_log_probs_all = old_log_probs_all[start:end].to(device)
        micro_old_log_probs_all = micro_old_log_probs_all[:, :policy_log_probs.size(1), :]

        loss, metadata = grpo_microbatch_train_step(
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,

            policy_log_probs=policy_log_probs,
            policy_log_probs_all=policy_log_probs_all,

            advantages=micro_advantages,
            old_log_probs=micro_old_log_probs,
            old_log_probs_all=micro_old_log_probs_all,

            kl_beta=kl_beta,
            clip_low=clip_low,
            clip_high=clip_high,
        )

        n_response_tokens = response_mask.sum().item()
        total_response_tokens += n_response_tokens
        total_loss += loss.detach().item() * gradient_accumulation_steps

        if "clip_fraction" in metadata:
            total_clip_fraction += float(metadata["clip_fraction"].item())
            clip_fraction_count += 1

        if "kl_mean" in metadata:
            total_kl_mean += float(metadata["kl_mean"].item())
            kl_mean_count += 1

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()

    current_lr = optimizer.param_groups[0]["lr"]

    metrics = {
        "loss": total_loss,
        "response_tokens": float(total_response_tokens),
        "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        "lr": float(current_lr),
    }

    if clip_fraction_count > 0:
        metrics["clip_fraction"] = total_clip_fraction / clip_fraction_count
    if kl_mean_count > 0:
        metrics["kl_mean"] = total_kl_mean / kl_mean_count

    return metrics


def save_latest_grpo_state(
    model,
    tokenizer,
    optimizer,
    scheduler,
    history,
    save_dir: str,
    step: int | None = None,
):
    """
    覆盖保存最新的：
    - model
    - tokenizer
    - optimizer state
    - scheduler state
    - history
    - RNG states（可选但推荐）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) 保存模型和 tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # 2) 保存 history
    history_path = os.path.join(save_dir, "grpo_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 3) 保存 optimizer / scheduler / 训练状态
    train_state = {
        "step": step,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(train_state, os.path.join(save_dir, "training_state.pt"))

    print(f"[Save latest] model/tokenizer/history/optimizer/scheduler saved to: {save_dir}")

### GRPO整体，n_grpo_steps次循环 -> sample questions -> 做G次pi_old的抽样得到prob_old和advantage ->进行epochs_per_rollout_batch次GRPO_step更新
def algorithm3_grpo(
    model,
    tokenizer,
    optimizer,
    scheduler,
    question_dataset: List[Dict[str, Any]],
    device,
    n_grpo_steps: int,
    rollout_batch_size: int,
    group_size: int,
    gradient_accumulation_steps: int,
    reward_fn=r1_zero_reward_fn,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int | None = None,
    max_grad_norm: float = 1.0,
    eval_every: int = 10,
    eval_test_path: str = "/home/u2022310886/jupyterlab/RL/math_lighteval_smalltest.jsonl",
    eval_batch_size: int = 8,
    history: list[dict] | None = None,
    curriculum_start_probs: dict[str, float] | None = None,
    curriculum_end_probs: dict[str, float] | None = None,
    save_dir: str | None = None,
    save_every: int = 50,
    kl_beta: float = 0.01,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
    old_cache_microbatch_size: int = 2,
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    if history is None:
        history = []

    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    if train_batch_size is None:
        train_batch_size = rollout_batch_size

    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先按 Level 分桶，只做一次
    level_buckets = build_level_buckets(question_dataset)

    start_step = len(history)

    for local_step in range(1, n_grpo_steps + 1):
        step = start_step + local_step
        step_start_time = time.perf_counter()

        # ========= 1. curriculum sample questions =========
        level_probs = get_curriculum_level_probs(
            local_step=local_step,
            n_grpo_steps=n_grpo_steps,
            start_probs=curriculum_start_probs,
            end_probs=curriculum_end_probs,
        )

        question_batch = sample_curriculum_batch(
            level_buckets=level_buckets,
            batch_size=n_prompts_per_rollout_batch,
            level_probs=level_probs,
        )

        prompt_once = [build_prompt(x["question"]) for x in question_batch]
        gold_once = [x["gold"] for x in question_batch]

        rollout_prompts = []
        repeated_ground_truths = []
        for p, g in zip(prompt_once, gold_once):
            rollout_prompts.extend([p] * group_size)
            repeated_ground_truths.extend([g] * group_size)

        # ========= 2. rollout with old policy =========
        model.eval()
        tokenizer.padding_side = "left"

        full_texts = generate_batch(
            prompts=rollout_prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            gen_cfg={
                "max_prompt_length": 1024,
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
            },
        )

        rollout_responses = []
        for full_text in full_texts:
            response = extract_assistant_response(full_text)
            response = truncate_to_answer_end(response)
            rollout_responses.append(response)

        tokenizer.padding_side = "right"

        # ========= 3. compute rewards / advantages =========
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        # ========= 4. cache old policy info =========
        cached_old_log_probs = None
        cached_old_log_probs_all = None

        with torch.inference_mode():
            old_log_probs_chunks = []
            old_log_probs_all_chunks = []
        
            for start in range(0, len(rollout_prompts), old_cache_microbatch_size):
                end = start + old_cache_microbatch_size
        
                micro_prompts = rollout_prompts[start:end]
                micro_responses = rollout_responses[start:end]
        
                tokenized = tokenize_prompt_and_output(
                    prompt_strs=micro_prompts,
                    output_strs=micro_responses,
                    tokenizer=tokenizer,
                )
                input_ids = tokenized["input_ids"].to(device)
                labels = tokenized["labels"].to(device)
                attention_mask = (input_ids != tokenizer.pad_token_id)
        
                old_out = get_response_distribution(
                    model=model,
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
        
                old_selected_log_probs = old_out["selected_log_probs"].detach().cpu()
                old_log_probs_all = old_out["log_probs_all"].detach().cpu()

                old_log_probs_chunks.append(old_selected_log_probs)
                old_log_probs_all_chunks.append(old_log_probs_all)

                del tokenized, input_ids, labels, attention_mask, old_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        cached_old_log_probs = pad_and_stack_2d(old_log_probs_chunks, pad_value=0.0)
        cached_old_log_probs_all = pad_and_stack_3d(old_log_probs_all_chunks, pad_value=0.0)

        # ========= 5. inner train loop =========
        assert train_batch_size % group_size == 0, (
            "train_batch_size must be divisible by group_size"
        )

        num_groups = rollout_batch_size // group_size
        group_indices = list(range(num_groups))
        step_metrics_accum = []

        for epoch in range(epochs_per_rollout_batch):
            # 只打乱“题目组”的顺序，不打乱组内样本顺序
            random.shuffle(group_indices)

            train_indices = []
            for g in group_indices:
                base = g * group_size
                train_indices.extend(range(base, base + group_size))

            for start in range(0, rollout_batch_size, train_batch_size):
                batch_idx = train_indices[start:start + train_batch_size]

                batch_prompts = [rollout_prompts[i] for i in batch_idx]
                batch_responses = [rollout_responses[i] for i in batch_idx]
                batch_advantages = advantages[batch_idx]

                batch_old_log_probs = None
                if cached_old_log_probs is not None:
                    batch_old_log_probs = cached_old_log_probs[batch_idx]


                batch_old_log_probs_all = None
                if cached_old_log_probs_all is not None:
                    batch_old_log_probs_all = cached_old_log_probs_all[batch_idx]

                train_metrics = grpo_step(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    prompt_strs=batch_prompts,
                    response_strs=batch_responses,
                    advantages=batch_advantages,
                    device=device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                
                    old_log_probs=batch_old_log_probs,
                    old_log_probs_all=batch_old_log_probs_all,
                
                    kl_beta=kl_beta,
                    clip_low=clip_low,
                    clip_high=clip_high,
                
                    max_grad_norm=max_grad_norm,
                )
                step_metrics_accum.append(train_metrics)

        avg_train_loss = sum(m["loss"] for m in step_metrics_accum) / max(len(step_metrics_accum), 1)
        avg_grad_norm = sum(m["grad_norm"] for m in step_metrics_accum) / max(len(step_metrics_accum), 1)
        avg_response_tokens = sum(m["response_tokens"] for m in step_metrics_accum) / max(len(step_metrics_accum), 1)
        avg_lr = sum(m.get("lr", 0.0) for m in step_metrics_accum) / max(len(step_metrics_accum), 1)

        metrics = {
            "grpo_step": step,
            "loss": avg_train_loss,
            "grad_norm": avg_grad_norm,
            "response_tokens": avg_response_tokens,
            "lr": avg_lr,
            "step_time_sec": time.perf_counter() - step_start_time,
            "train_reward_mean": float(reward_meta["reward_mean"].item()),
            "train_reward_std": float(reward_meta["reward_std"].item()),
            "train_format_reward_mean": float(reward_meta["format_reward_mean"].item()),
            "sample_prob_level_1": level_probs.get("Level 1", 0.0),
            "sample_prob_level_2": level_probs.get("Level 2", 0.0),
            "sample_prob_level_3": level_probs.get("Level 3", 0.0),
            "sample_prob_level_4": level_probs.get("Level 4", 0.0),
            "sample_prob_level_5": level_probs.get("Level 5", 0.0),
        }

        if len(raw_rewards) > 0:
            metrics["train_answer_reward_mean"] = float(raw_rewards.float().mean().item())
        if len(step_metrics_accum) > 0 and any("kl_mean" in m for m in step_metrics_accum):
            metrics["kl_mean"] = (
                sum(m.get("kl_mean", 0.0) for m in step_metrics_accum)
                / max(len(step_metrics_accum), 1)
            )
        sampled_level_count = {}
        for ex in question_batch:
            lvl = ex.get("Level", ex.get("level", "Level 3"))
            sampled_level_count[lvl] = sampled_level_count.get(lvl, 0) + 1

        for lvl in LEVEL_NAMES:
            metrics[f"sampled_{lvl.replace(' ', '_').lower()}_count"] = sampled_level_count.get(lvl, 0)

        if len(step_metrics_accum) > 0 and "clip_fraction" in step_metrics_accum[0]:
            metrics["clip_fraction"] = (
                sum(m.get("clip_fraction", 0.0) for m in step_metrics_accum)
                / max(len(step_metrics_accum), 1)
            )

        print(
            f"[GRPO step {step}] "
            f"loss={metrics['loss']:.4f}, "
            f"reward={metrics['train_reward_mean']:.4f}, "
            f"kl={metrics.get('kl_mean', 0.0):.4f}, "
            f"clip={metrics.get('clip_fraction', 0.0):.4f}, "
            f"lr={metrics['lr']:.2e}, "
            f"time={metrics['step_time_sec']:.2f}s"
        )

        history.append(metrics)
        if save_dir is not None and save_every > 0 and step % save_every == 0:
            save_latest_grpo_state(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                save_dir=save_dir,
                step=step,
            )
    if save_dir is not None:
        save_latest_grpo_state(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            save_dir=save_dir,
            step=(len(history) if history is not None else None),
        )
    return model, history


def main():
    # ===== 路径配置 =====
    model_path = "/home/u2022310886/jupyterlab/RL/models/Qwen2.5-Math-1.5B-SFT-LoRA-Merged"
    save_dir = "/home/u2022310886/jupyterlab/RL/models/Qwen2.5-Math-1.5B-SFT-FullGRPO-oldclip-pureKL"
    train_path = "/home/u2022310886/jupyterlab/RL/math_lighteval_train.jsonl"
    training_state_path = os.path.join(save_dir, "training_state.pt")

    # ===== 超参数 =====
    n_grpo_steps = 1000
    rollout_batch_size = 32
    group_size = 8
    train_batch_size = 8
    gradient_accumulation_steps = 8
    lr = 1e-5
    lr_min = 5e-6
    warmup_ratio = 0.08
    max_grad_norm = 1.0
    seed = 44

    kl_beta = 0.5
    clip_low = 0.8
    clip_high = 1.2
    old_cache_microbatch_size = 4
    
    normalize_by_std = False
    advantage_eps = 1e-6
    epochs_per_rollout_batch = 4

    eval_every = 10
    eval_test_path = "/home/u2022310886/jupyterlab/RL/math_lighteval_smalltest.jsonl"
    eval_batch_size = 8

    curriculum_start_probs = {
        "Level 1": 0.35,
        "Level 2": 0.30,
        "Level 3": 0.20,
        "Level 4": 0.10,
        "Level 5": 0.05,
    }
    curriculum_end_probs = {
        "Level 1": 0.10,
        "Level 2": 0.15,
        "Level 3": 0.20,
        "Level 4": 0.25,
        "Level 5": 0.30,
    }

    # ===== 随机种子 =====
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ===== 设备 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 读取数据 =====
    question_dataset = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("question") is None or rec.get("gold") is None:
                continue
            question_dataset.append(rec)

    print(f"Loaded {len(question_dataset)} GRPO examples from: {train_path}")
    if len(question_dataset) == 0:
        raise ValueError("GRPO dataset is empty.")

    # ===== 决定从哪里恢复 adapter =====
    history_path = os.path.join(save_dir, "grpo_history.json")
    resume_model_path = save_dir if os.path.exists(history_path) else model_path

    print(f"Resume model path: {resume_model_path}")
    
    print(f"Loading tokenizer from: {resume_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        resume_model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading full model from: {resume_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        resume_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    
    model.to(device)
    model.config.use_cache = False

    # ===== 检查可训练参数 =====
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"total_params: {total_params}")
    print(f"trainable_params: {trainable_params}")
    print(f"trainable_ratio: {100 * trainable_params / total_params:.6f}%")

    if trainable_params == 0:
        raise ValueError("No trainable parameters found.")

    # ===== 读取已有 history =====
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        print(f"Loaded existing history from: {history_path}")
        print(f"Existing history length: {len(history)}")
    else:
        history = []
        print("No existing history found. Start with empty history.")

    # ===== 优化器：只优化 requires_grad=True 的参数 =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )
    
    completed_steps = len(history)
    remaining_steps = max(n_grpo_steps - completed_steps, 0)
    
    updates_per_grpo_step = epochs_per_rollout_batch * (rollout_batch_size // train_batch_size)
    total_optimizer_steps = n_grpo_steps * updates_per_grpo_step
    completed_optimizer_steps = completed_steps * updates_per_grpo_step
    remaining_optimizer_steps = max(total_optimizer_steps - completed_optimizer_steps, 1)
    
    warmup_steps_total = max(1, int(total_optimizer_steps * warmup_ratio))
    remaining_warmup_steps = max(warmup_steps_total - completed_optimizer_steps, 0)
    
    scheduler = build_scheduler(
        optimizer=optimizer,
        warmup_steps=remaining_warmup_steps,
        total_steps=remaining_optimizer_steps,
        lr_min=lr_min,
    )
    
    # ===== 恢复 optimizer / scheduler / RNG 状态 =====
    if os.path.exists(training_state_path):
        print(f"Loading training state from: {training_state_path}")
        training_state = torch.load(training_state_path, map_location="cpu")
    
        if training_state.get("optimizer") is not None:
            optimizer.load_state_dict(training_state["optimizer"])
            print("Optimizer state restored.")
    
        if scheduler is not None and training_state.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(training_state["scheduler"])
                print("Scheduler state restored.")
            except Exception as e:
                print(f"Warning: failed to restore scheduler state: {e}")
                print("Will continue with rebuilt scheduler.")
    
        # 恢复随机状态（可选但推荐）
        if "random_state" in training_state and training_state["random_state"] is not None:
            random.setstate(training_state["random_state"])
            print("Python random state restored.")
    
        if "torch_rng_state" in training_state and training_state["torch_rng_state"] is not None:
            torch.set_rng_state(training_state["torch_rng_state"])
            print("Torch RNG state restored.")
    
        if (
            torch.cuda.is_available()
            and "cuda_rng_state_all" in training_state
            and training_state["cuda_rng_state_all"] is not None
        ):
            torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
            print("CUDA RNG state restored.")
    else:
        print("No training_state.pt found. Optimizer/scheduler start fresh.")

    print(f"Completed steps: {completed_steps}")
    print(f"Remaining steps: {remaining_steps}")
    print(f"updates_per_grpo_step: {updates_per_grpo_step}")
    print(f"total_optimizer_steps: {total_optimizer_steps}")
    print(f"completed_optimizer_steps: {completed_optimizer_steps}")
    print(f"remaining_optimizer_steps: {remaining_optimizer_steps}")
    print(f"warmup_steps_total: {warmup_steps_total}")
    print(f"remaining_warmup_steps: {remaining_warmup_steps}")

    if remaining_steps == 0:
        print("Target n_grpo_steps already reached. No further training needed.")
        return

    model, history = algorithm3_grpo(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        question_dataset=question_dataset,
        device=device,
        n_grpo_steps=remaining_steps,
        rollout_batch_size=rollout_batch_size,
        group_size=group_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        reward_fn=r1_zero_reward_fn,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
        epochs_per_rollout_batch=epochs_per_rollout_batch,
        train_batch_size=train_batch_size,
        max_grad_norm=max_grad_norm,
        eval_every=eval_every,
        eval_test_path=eval_test_path,
        eval_batch_size=eval_batch_size,
        history=history,
        curriculum_start_probs=curriculum_start_probs,
        curriculum_end_probs=curriculum_end_probs,
        save_dir=save_dir,
        save_every=50,
        kl_beta=kl_beta,
        clip_low=clip_low,
        clip_high=clip_high,
        old_cache_microbatch_size=old_cache_microbatch_size,
    )

    # save_latest_grpo_state(
    #     model=model,
    #     tokenizer=tokenizer,
    #     history=history,
    #     save_dir=save_dir,
    # )

    print("GRPO finished.")
    print(f"Final history length: {len(history)}")


if __name__ == "__main__":
    main()