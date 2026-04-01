# GRPO.py
import os
import json
import time
import math
import random
from typing import List, Dict, Any, Literal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from baseline_funs import build_prompt, extract_assistant_response, generate_batch
from drgrpo_grader_new import r1_zero_reward_fn
from SFT import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    masked_normalize,
    truncate_to_answer_end,
)

from collections import defaultdict

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
### 没有clip的情况
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages,
    policy_log_probs,
):
    """
    batch_size = n_groups * group_size * steps_per_rollout_batch
    raw_rewards_or_advantages: shape: (batch_size, 1)
    policy_log_probs: shape: (batch_size, seq_length)
    """
    return -(raw_rewards_or_advantages * policy_log_probs)

### clip的情况的目标函数
def compute_grpo_clip_loss(
    advantages,
    policy_log_probs,
    old_log_probs,
    cliprange,
):
    """
    advantages: (batch_size, 1)
    policy_log_probs: (batch_size, seq_length)
    old_log_probs: (batch_size, seq_length)
    """
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)

    unclipped_obj = advantages * ratios
    clipped_obj = advantages * clipped_ratios

    per_token_obj = torch.min(unclipped_obj, clipped_obj)
    loss = -per_token_obj

    token_was_clipped = ratios != clipped_ratios

    metadata = {
        "clip_fraction": token_was_clipped.float().mean().detach(),
    }

    return loss, metadata

### 目标函数汇总
def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
):
    """
    Return:
        loss: (batch_size, seq_length)
        metadata: dict[str, torch.Tensor]
    """
    metadata = {}

    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards is required for no_baseline"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        )
        metadata["loss_type"] = "no_baseline"

    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages is required for reinforce_with_baseline"
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
        metadata["loss_type"] = "reinforce_with_baseline"

    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages is required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs is required for grpo_clip"
        assert cliprange is not None, "cliprange is required for grpo_clip"

        loss, clip_meta = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
        metadata["loss_type"] = "grpo_clip"
        metadata.update(clip_meta)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

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

    return res

### 一个microbatch中的GRPO更新
def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on one GRPO microbatch.
    """
    per_token_loss, loss_meta = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    # 每条样本在 response token 上求平均
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)

    # 再对 batch 求和
    loss = per_example_loss.sum()

    # gradient accumulation
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = {}
    metadata.update(loss_meta)
    return loss, metadata

###一次GRPOstep, 包括gradient_accumulation_steps次grpo_microbatch_train_step
def grpo_step(
    model,
    tokenizer,
    optimizer,
    scheduler,
    prompt_strs: list[str],
    response_strs: list[str],
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    device: torch.device,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """
    对一个 rollout batch 做一次 optimizer update。
    """
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
        attention_mask = input_ids != pad_token_id

        forward_out = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            return_token_entropy=False,
        )

        policy_log_probs = forward_out["log_probs"]

        micro_raw_rewards = raw_rewards[start:end].to(device).unsqueeze(1)
        micro_advantages = advantages[start:end].to(device).unsqueeze(1)

        micro_old_log_probs = None
        if old_log_probs is not None:
            micro_old_log_probs = old_log_probs[start:end].to(device)
            micro_old_log_probs = micro_old_log_probs[:, :policy_log_probs.size(1)]

            assert micro_old_log_probs.shape == policy_log_probs.shape, (
                f"Shape mismatch after slicing: "
                f"policy_log_probs={policy_log_probs.shape}, "
                f"micro_old_log_probs={micro_old_log_probs.shape}"
            )

        loss, metadata = grpo_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            loss_type=loss_type,
            raw_rewards=micro_raw_rewards,
            advantages=micro_advantages,
            old_log_probs=micro_old_log_probs,
            cliprange=cliprange,
        )

        n_response_tokens = response_mask.sum().item()
        total_response_tokens += n_response_tokens
        total_loss += loss.detach().item() * gradient_accumulation_steps

        if "clip_fraction" in metadata:
            total_clip_fraction += float(metadata["clip_fraction"].item())
            clip_fraction_count += 1

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()

    current_lr = optimizer.param_groups[0]["lr"]

    metrics = {
        "loss": total_loss,
        "response_tokens": float(total_response_tokens),
        "avg_token_entropy": 0.0,
        "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        "lr": float(current_lr),
    }

    if clip_fraction_count > 0:
        metrics["clip_fraction"] = total_clip_fraction / clip_fraction_count

    return metrics


def save_latest_grpo_state(
    model,
    tokenizer,
    history,
    save_dir: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
):
    """
    覆盖保存最新的模型/tokenizer，
    并保存完整累积的 GRPO history。
    额外保存 optimizer / scheduler 状态，便于断点续训。
    """
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    history_path = os.path.join(save_dir, "grpo_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if optimizer is not None:
        optimizer_state_path = os.path.join(save_dir, "optimizer.pt")
        torch.save(optimizer.state_dict(), optimizer_state_path)

    if scheduler is not None:
        scheduler_state_path = os.path.join(save_dir, "scheduler.pt")
        torch.save(scheduler.state_dict(), scheduler_state_path)

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
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline",
    reward_fn=r1_zero_reward_fn,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    cliprange: float = 0.2,
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

    level_buckets = build_level_buckets(question_dataset)
    start_step = len(history)

    for local_step in range(1, n_grpo_steps + 1):
        step = start_step + local_step
        step_start_time = time.perf_counter()

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

        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        cached_old_log_probs = None
        if loss_type == "grpo_clip":
            with torch.inference_mode():
                old_logprob_microbatch_size = 4
                old_log_probs_chunks = []

                for start in range(0, len(rollout_prompts), old_logprob_microbatch_size):
                    end = start + old_logprob_microbatch_size

                    micro_prompts = rollout_prompts[start:end]
                    micro_responses = rollout_responses[start:end]

                    tokenized = tokenize_prompt_and_output(
                        prompt_strs=micro_prompts,
                        output_strs=micro_responses,
                        tokenizer=tokenizer,
                    )
                    input_ids = tokenized["input_ids"].to(device)
                    labels = tokenized["labels"].to(device)
                    attention_mask = input_ids != tokenizer.pad_token_id

                    old_out = get_response_log_probs(
                        model=model,
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        return_token_entropy=False,
                    )

                    old_log_probs_chunks.append(old_out["log_probs"].detach().cpu())

                    del tokenized, input_ids, labels, attention_mask, old_out
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                max_seq_len = max(x.size(1) for x in old_log_probs_chunks)
                padded_chunks = []

                for x in old_log_probs_chunks:
                    if x.size(1) < max_seq_len:
                        pad_width = max_seq_len - x.size(1)
                        x = torch.nn.functional.pad(x, (0, pad_width), mode="constant", value=0.0)
                    padded_chunks.append(x)

                cached_old_log_probs = torch.cat(padded_chunks, dim=0)

        assert train_batch_size % group_size == 0, (
            "train_batch_size must be divisible by group_size"
        )

        num_groups = rollout_batch_size // group_size
        group_indices = list(range(num_groups))
        step_metrics_accum = []

        for epoch in range(epochs_per_rollout_batch):
            random.shuffle(group_indices)

            train_indices = []
            for g in group_indices:
                base = g * group_size
                train_indices.extend(range(base, base + group_size))

            for start in range(0, rollout_batch_size, train_batch_size):
                batch_idx = train_indices[start:start + train_batch_size]

                batch_prompts = [rollout_prompts[i] for i in batch_idx]
                batch_responses = [rollout_responses[i] for i in batch_idx]
                batch_raw_rewards = raw_rewards[batch_idx]
                batch_advantages = advantages[batch_idx]

                batch_old_log_probs = None
                if cached_old_log_probs is not None:
                    batch_old_log_probs = cached_old_log_probs[batch_idx]

                train_metrics = grpo_step(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    prompt_strs=batch_prompts,
                    response_strs=batch_responses,
                    raw_rewards=batch_raw_rewards,
                    advantages=batch_advantages,
                    device=device,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    old_log_probs=batch_old_log_probs,
                    cliprange=cliprange,
                    max_grad_norm=max_grad_norm,
                )
                step_metrics_accum.append(train_metrics)

        avg_train_loss = sum(m["loss"] for m in step_metrics_accum) / max(len(step_metrics_accum), 1)
        avg_grad_norm = sum(m["grad_norm"] for m in step_metrics_accum) / max(len(step_metrics_accum), 1)
        avg_entropy = sum(m["avg_token_entropy"] for m in step_metrics_accum) / max(len(step_metrics_accum), 1)
        avg_response_tokens = sum(m["response_tokens"] for m in step_metrics_accum) / max(len(step_metrics_accum), 1)
        avg_lr = sum(m.get("lr", 0.0) for m in step_metrics_accum) / max(len(step_metrics_accum), 1)

        metrics = {
            "grpo_step": step,
            "loss": avg_train_loss,
            "grad_norm": avg_grad_norm,
            "avg_token_entropy": avg_entropy,
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
            f"entropy={metrics['avg_token_entropy']:.4f}, "
            f"lr={metrics['lr']:.2e}, "
            f"L1={metrics['sample_prob_level_1']:.2f}, "
            f"L5={metrics['sample_prob_level_5']:.2f}, "
            f"time={metrics['step_time_sec']:.2f}s"
        )

        history.append(metrics)
        if save_dir is not None and save_every > 0 and step % save_every == 0:
            save_latest_grpo_state(
                model=model,
                tokenizer=tokenizer,
                history=history,
                save_dir=save_dir,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    if save_dir is not None:
        save_latest_grpo_state(
            model=model,
            tokenizer=tokenizer,
            history=history,
            save_dir=save_dir,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    return model, history


def main():
    # ===== 路径配置 =====
    model_path = "/home/u2022310886/jupyterlab/RL/models/Qwen2.5-Math-1.5B-SFT-LoRA-Merged"
    save_dir = "/home/u2022310886/jupyterlab/RL/models/Qwen2.5-Math-1.5B-SFT-FullGRPO_step_1000"
    train_path = "/home/u2022310886/jupyterlab/RL/math_lighteval_train.jsonl"

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

    loss_type = "grpo_clip"
    normalize_by_std = False
    advantage_eps = 1e-6
    cliprange = 0.2
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

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    history_path = os.path.join(save_dir, "grpo_history.json")
    optimizer_path = os.path.join(save_dir, "optimizer.pt")
    scheduler_path = os.path.join(save_dir, "scheduler.pt")
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"total_params: {total_params}")
    print(f"trainable_params: {trainable_params}")
    print(f"trainable_ratio: {100 * trainable_params / total_params:.6f}%")

    if trainable_params == 0:
        raise ValueError("No trainable parameters found.")

    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        print(f"Loaded existing history from: {history_path}")
        print(f"Existing history length: {len(history)}")
    else:
        history = []
        print("No existing history found. Start with empty history.")

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

    if os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        print(f"Loaded optimizer state from: {optimizer_path}")
    else:
        print("No existing optimizer state found. Start with fresh optimizer.")

    if os.path.exists(scheduler_path):
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=device))
        print(f"Loaded scheduler state from: {scheduler_path}")
    else:
        print("No existing scheduler state found. Start with fresh scheduler.")

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
        loss_type=loss_type,
        reward_fn=r1_zero_reward_fn,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
        cliprange=cliprange,
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
    )

    print("GRPO finished.")
    print(f"Final history length: {len(history)}")


if __name__ == "__main__":
    main()
