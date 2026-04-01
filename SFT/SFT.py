import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
import math
import os
import json
import random
from typing import List, Dict, Any
from pathlib import Path
from drgrpo_grader_new import r1_zero_reward_fn
from baseline_funs import generate_batch, build_prompt, extract_assistant_response
import time

### 将prompt和output打包好
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    """
    Tokenize prompt and output separately, concatenate them, and construct
    a response mask aligned with `labels`.

    Returns:
        dict with keys:
            input_ids:      (batch_size, max_len - 1)
            labels:         (batch_size, max_len - 1)
            response_mask:  (batch_size, max_len - 1)
    """
    assert len(prompt_strs) == len(output_strs)

    prompt_tokenized = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    output_tokenized = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )

    full_input_ids = []
    response_masks = []

    for prompt_ids, output_ids in zip(prompt_tokenized["input_ids"], output_tokenized["input_ids"]):
        full_ids = prompt_ids + output_ids

        # 至少要有两个 token，才能构造 input_ids 和 labels
        if len(full_ids) < 2:
            raise ValueError("Each prompt+output pair must contain at least 2 tokens.")

        full_input_ids.append(full_ids)

        # response_mask 与 labels 对齐，长度应为 len(full_ids) - 1
        # labels = full_ids[1:]
        # 其中属于 output 的 labels 位置为 [len(prompt_ids)-1, ..., len(full_ids)-2]
        mask = [0] * (len(prompt_ids) - 1) + [1] * len(output_ids)
        response_masks.append(mask)

    max_len = max(len(x) for x in full_input_ids)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id.")

    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    for full_ids, mask in zip(full_input_ids, response_masks):
        # 先 padding 到 max_len
        padded_full_ids = full_ids + [pad_token_id] * (max_len - len(full_ids))

        # input_ids / labels
        input_ids = padded_full_ids[:-1]
        labels = padded_full_ids[1:]

        # mask 本身长度就是 len(full_ids)-1，还需 pad 到 max_len-1
        padded_mask = mask + [0] * ((max_len - 1) - len(mask))

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)
        batch_response_mask.append(padded_mask)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "response_mask": torch.tensor(batch_response_mask, dtype=torch.bool),
    }

### 计算交叉熵
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy over the vocabulary dimension.

    Args:
        logits: Tensor of shape (batch_size, sequence_length, vocab_size)

    Returns:
        Tensor of shape (batch_size, sequence_length)
    """
    log_probs = torch.log_softmax(logits, dim=-1)   # (B, T, V)
    probs = torch.exp(log_probs)                    # (B, T, V)
    entropy = -(probs * log_probs).sum(dim=-1)     # (B, T)
    return entropy
    
### 计算条件概率————要记得输入attention_mask
# 示例
# ```python
# attention_mask = (input_ids != pad_token_id)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask=None,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities for `labels` given `input_ids`.
    
    Args:
        model: HuggingFace causal language model.
        input_ids: (batch_size, sequence_length)
        labels: (batch_size, sequence_length)
        return_token_entropy: whether to also return per-token entropy.

    Returns:
        dict with:
            "log_probs": (batch_size, sequence_length)
            optionally "token_entropy": (batch_size, sequence_length)
    """

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).logits
    log_probs_all = torch.log_softmax(logits, dim=-1)    # (B, T, V)

    log_probs = torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)                                        # (B, T)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result

### 计算损失函数————只计算response部分

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over tensor elements selected by mask, then divide by a fixed constant.

    Args:
        tensor: Tensor to be summed.
        mask: Same shape as tensor; True/1 means include, False/0 means ignore.
        normalize_constant: Constant used to normalize the masked sum.
        dim: Dimension to sum over. If None, sum over all elements.

    Returns:
        Normalized masked sum.
    """
    masked_sum = (tensor * mask).sum(dim=dim)
    return masked_sum / normalize_constant

### SFT计算一次损失函数，要计算gradient_accumulation_steps次才迭代一次

# 示例：

# ```python
# optimizer.zero_grad()

# for i in range(4):
#     loss, metadata = sft_microbatch_train_step(...)
#     # 这里只 backward，不 step

# optimizer.step()
# optimizer.zero_grad()
def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on one SFT microbatch.

    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probs
        response_mask: (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding
        gradient_accumulation_steps: number of microbatches per optimizer step
        normalize_constant: constant used in masked normalization

    Returns:
        loss: scalar tensor, already divided by gradient_accumulation_steps
        metadata: dict of useful logging stats
    """
    token_losses = -policy_log_probs
    loss = masked_normalize(
        tensor=token_losses,
        mask=response_mask,
        normalize_constant=normalize_constant
    )
    loss = loss / gradient_accumulation_steps
    loss.backward()

    metadata = {
        'loss': loss.detach(),
    }
    return loss, metadata

### SFT训练函数
def sft_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    prompt_strs: list[str],
    output_strs: list[str],
    device: torch.device,
    gradient_accumulation_steps: int,
    max_grad_norm: float = 1.0,
    pad_token_id: int | None = None,
) -> dict[str, float]:
    """
    Execute one optimizer update for SFT with gradient accumulation.

    Args:
        model: causal LM
        tokenizer: tokenizer
        optimizer: optimizer
        prompt_strs: full batch prompts
        output_strs: full batch target outputs
        device: cuda / cpu device
        gradient_accumulation_steps: number of microbatches per optimizer step
        max_grad_norm: gradient clipping threshold
        pad_token_id: optional override for pad token id

    Returns:
        dict of logging metrics
    """
    assert len(prompt_strs) == len(output_strs), "prompt/output batch size mismatch"
    batch_size = len(prompt_strs)
    assert batch_size > 0, "empty batch is not allowed"
    assert batch_size % gradient_accumulation_steps == 0, (
        "For this implementation, batch_size must be divisible by gradient_accumulation_steps"
    )

    model.train()
    optimizer.zero_grad()

    microbatch_size = batch_size // gradient_accumulation_steps

    total_loss = 0.0
    total_response_tokens = 0
    # total_entropy = 0.0

    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id.")

    for micro_idx in range(gradient_accumulation_steps):
        start = micro_idx * microbatch_size
        end = (micro_idx + 1) * microbatch_size

        micro_prompts = prompt_strs[start:end]
        micro_outputs = output_strs[start:end]

        batch = tokenize_prompt_and_output(
            prompt_strs=micro_prompts,
            output_strs=micro_outputs,
            tokenizer=tokenizer,
        )

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        response_mask = batch["response_mask"].to(device)

        attention_mask = (input_ids != pad_token_id)

        forward_out = get_response_log_probs(
            model=model,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            return_token_entropy=False,
        )

        policy_log_probs = forward_out["log_probs"]
        token_entropy = None

        n_response_tokens = response_mask.sum().item()

        loss, metadata = sft_microbatch_train_step(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            gradient_accumulation_steps=gradient_accumulation_steps,
            normalize_constant=max(n_response_tokens, 1),
        )

        # 日志统计：这里把“已经除过 grad_acc_steps 的 loss”还原成更直观的累计值
        total_loss += loss.detach().item() * gradient_accumulation_steps

        total_response_tokens += n_response_tokens

        ### 先暂时不计算entropy
        # avg_entropy_this_micro = masked_normalize(
        #     tensor=token_entropy.detach(),
        #     mask=response_mask,
        #     normalize_constant=max(n_response_tokens, 1),
        # ).item()
        # total_entropy += avg_entropy_this_micro * n_response_tokens

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    # avg_token_entropy = total_entropy / max(total_response_tokens, 1)

    metrics = {
        "loss": total_loss,  # 这是每次 optimizer update 的累计 microbatch loss
        "response_tokens": float(total_response_tokens),
        #"avg_token_entropy": 0.0,
        "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
    }
    return metrics

### Algorithm 1
def sample_sft_batch(
    sft_dataset: List[Dict[str, str]],
    batch_size: int,
) -> List[Dict[str, str]]:
    """从 SFT 数据集中随机采样一个 batch。"""
    assert len(sft_dataset) > 0, "sft_dataset 不能为空"
    if batch_size <= len(sft_dataset):
        return random.sample(sft_dataset, batch_size)
    # 若 batch_size 比数据集大，则允许有放回采样
    return [random.choice(sft_dataset) for _ in range(batch_size)]

### 截断函数
def truncate_to_answer_end(text: str) -> str:
    """
    截断到第一个 </answer> 为止，并保留 </answer>。
    如果没有找到，则原样返回。
    """
    end_tag = "</answer>"
    idx = text.find(end_tag)
    if idx == -1:
        return text
    return text[: idx + len(end_tag)]

### 测评函数
@torch.no_grad()
def evaluate_smalltest_rewards(
    model,
    tokenizer,
    device,
    test_path: str,
    batch_size: int = 8,
    gen_cfg: dict | None = None,
):
    """
    在 small test 集上评估平均 format_reward / answer_reward / reward
    数据格式默认每条至少包含:
        {
            "question": ...,
            "gold": ...
        }
    """
    gen_cfg = gen_cfg or {
        "max_prompt_length": 1024,
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("gold") is None:
                continue
            samples.append(rec)

    if len(samples) == 0:
        raise ValueError(f"No valid samples found in {test_path}")

    model.eval()
    tokenizer.padding_side = "left"
    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_reward = 0.0
    total_count = 0
    sample_idx = 0
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        prompts = [build_prompt(x["question"]) for x in batch]

        full_texts = generate_batch(
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            gen_cfg=gen_cfg,
        )

        for rec, full_text in zip(batch, full_texts):
            sample_idx += 1
            if sample_idx % 10 == 0 or sample_idx == len(samples):
                print(f"[Eval] processed {sample_idx}/{len(samples)} samples")
            # 只保留 Assistant 的回答部分
            response = extract_assistant_response(full_text)
        
            # 截断到第一个 </answer>
            response = truncate_to_answer_end(response)
        
            reward_out = r1_zero_reward_fn(response, rec["gold"])

            total_format_reward += float(reward_out.get("format_reward", 0.0))
            total_answer_reward += float(reward_out.get("answer_reward", 0.0))
            total_reward += float(reward_out.get("reward", 0.0))
            total_count += 1

    metrics = {
        "eval_num_examples": total_count,
        "eval_avg_format_reward": total_format_reward / max(total_count, 1),
        "eval_avg_answer_reward": total_answer_reward / max(total_count, 1),
        "eval_avg_reward": total_reward / max(total_count, 1),
    }
    tokenizer.padding_side = "right"
    model.train()
    return metrics

### 临时保存结果
def save_latest_sft_state(model, tokenizer, history, save_dir: str):
    """
    覆盖保存最新的模型/tokenizer，
    并保存完整累积的 history（不丢之前记录）。
    """
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    history_path = os.path.join(save_dir, "sft_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[Save latest] model/tokenizer/history saved to: {save_dir}")

def algorithm1_sft(
    model,
    tokenizer,
    optimizer,
    sft_dataset: List[Dict[str, str]],
    device,
    n_sft_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_grad_norm: float = 1.0,
    save_dir: str | None = None,
    eval_every: int = 50,
    eval_test_path: str = "/home/u2022310886/jupyterlab/RL/math_lighteval_smalltest.jsonl",
    eval_batch_size: int = 8,
    history: list[dict] | None = None,
):
    if history is None:
        history = []

    start_step = len(history)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for local_step in range(1, n_sft_steps + 1):
        step = start_step + local_step
        step_start_time = time.perf_counter()

        batch = sample_sft_batch(sft_dataset, batch_size)

        prompt_strs = [build_prompt(x["question"]) for x in batch]
        output_strs = [x["response"] for x in batch]

        metrics = sft_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            prompt_strs=prompt_strs,
            output_strs=output_strs,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
        )

        step_time_sec = time.perf_counter() - step_start_time
        metrics["sft_step"] = step
        metrics["step_time_sec"] = step_time_sec

        print(
            f"[SFT step {step}] "
            f"loss={metrics['loss']:.4f}, "
            f"resp_tokens={metrics['response_tokens']:.0f}, "
            f"grad_norm={metrics['grad_norm']:.4f}, "
            f"time={metrics['step_time_sec']:.2f}s"
        )

        if eval_every > 0 and step % eval_every == 0:
            eval_metrics = evaluate_smalltest_rewards(
                model=model,
                tokenizer=tokenizer,
                device=device,
                test_path=eval_test_path,
                batch_size=eval_batch_size,
                gen_cfg={
                    "max_prompt_length": 1024,
                    "max_new_tokens": 1024,
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
            )

            metrics.update(eval_metrics)

            print(
                f"[Eval @ step {step}] "
                f"format_reward={eval_metrics['eval_avg_format_reward']:.4f}, "
                f"answer_reward={eval_metrics['eval_avg_answer_reward']:.4f}, "
                f"reward={eval_metrics['eval_avg_reward']:.4f}, "
                f"n={eval_metrics['eval_num_examples']}"
            )

            history.append(metrics)

            if save_dir is not None:
                save_latest_sft_state(
                    model=model,
                    tokenizer=tokenizer,
                    history=history,
                    save_dir=save_dir,
                )
        else:
            history.append(metrics)

    if save_dir is not None:
        save_latest_sft_state(
            model=model,
            tokenizer=tokenizer,
            history=history,
            save_dir=save_dir,
        )

    return model, history

def main():
    import os
    import json
    import random
    import torch

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # ===== 路径配置 =====
    model_path = "/home/u2022310886/jupyterlab/RL/models"
    train_path = "/home/u2022310886/jupyterlab/RL/math_lighteval_train_fitted.jsonl"
    save_dir = "/home/u2022310886/jupyterlab/RL/models/Qwen2.5-Math-1.5B-SFT"

    # ===== 超参数 =====
    n_sft_steps = 1000
    batch_size = 16
    gradient_accumulation_steps = 16
    lr = 1e-5
    max_grad_norm = 1
    seed = 44

    # ===== 随机种子 =====
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ===== 设备 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 读取数据 =====
    sft_dataset = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sft_dataset.append(json.loads(line))

    print(f"Loaded {len(sft_dataset)} SFT examples from: {train_path}")
    if len(sft_dataset) == 0:
        raise ValueError("SFT dataset is empty.")

    # ===== 决定从哪里恢复模型 =====
    history_path = os.path.join(save_dir, "sft_history.json")
    resume_model_path = save_dir if os.path.exists(history_path) else model_path

    print(f"Resume model path: {resume_model_path}")

    # ===== 加载 tokenizer / model =====
    print(f"Loading tokenizer from: {resume_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(resume_model_path, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from: {resume_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        resume_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.config.use_cache = False

    # ===== 读取已有 history，用于续写 =====
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        print(f"Loaded existing history from: {history_path}")
        print(f"Existing history length: {len(history)}")
        if len(history) > 0 and "sft_step" in history[-1]:
            print(f"Last recorded sft_step: {history[-1]['sft_step']}")
    else:
        history = []
        print("No existing history found. Start with empty history.")

    # ===== 优化器 =====
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ===== 启动 SFT =====
    completed_steps = len(history)
    remaining_steps = max(n_sft_steps - completed_steps, 0)
    
    print(f"Completed steps: {completed_steps}")
    print(f"Remaining steps: {remaining_steps}")
    
    if remaining_steps == 0:
        print("Target n_sft_steps already reached. No further training needed.")
        return
    model, history = algorithm1_sft(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        sft_dataset=sft_dataset,
        device=device,
        n_sft_steps=remaining_steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        save_dir=save_dir,
        eval_every=50,
        eval_test_path="/home/u2022310886/jupyterlab/RL/math_lighteval_smalltest.jsonl",
        eval_batch_size=8,
        history=history,
    )

    print("SFT finished.")
    print(f"Final history length: {len(history)}")

if __name__ == "__main__":
    main()