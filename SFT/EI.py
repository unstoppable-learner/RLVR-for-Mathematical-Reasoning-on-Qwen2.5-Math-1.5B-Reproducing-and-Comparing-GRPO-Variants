import copy
from typing import Callable, List, Dict, Any
import random
import torch

###抽batch_size个问题和答案
def sample_question_batch(
    question_dataset: List[Dict[str, Any]],
    batch_size: int,
) -> List[Dict[str, Any]]:
    """
    question_dataset 形如:
    [{"question": ..., "gold": ...}, ...]
    """
    assert len(question_dataset) > 0, "question_dataset 不能为空"
    if batch_size <= len(question_dataset):
        return random.sample(question_dataset, batch_size)
    return [random.choice(question_dataset) for _ in range(batch_size)]

##过滤出reward==1的问答
from baseline_funs import build_prompt
def build_sft_dataset_from_rollouts(
    question_batch,
    rollout_outputs_per_question,
    reward_fn,
):
    filtered_sft_dataset = []

    total_rollouts = 0
    n_correct = 0
    n_format_ok = 0

    for rec, responses in zip(question_batch, rollout_outputs_per_question):
        question = rec["question"]
        gold = rec["solution"]

        # 直接复用你 baseline_funs.py 里的 prompt
        prompt = build_prompt(question)

        for response in responses:
            total_rollouts += 1
            reward_info = reward_fn(response, gold)

            if reward_info["format_reward"] == 1.0:
                n_format_ok += 1

            if reward_info["reward"] == 1.0:
                n_correct += 1
                filtered_sft_dataset.append({
                    "prompt": prompt,
                    "response": response,
                })

    stats = {
        "total_rollouts": total_rollouts,
        "n_format_ok": n_format_ok,
        "n_correct": n_correct,
        "keep_ratio": n_correct / max(total_rollouts, 1),
        "format_ok_ratio": n_format_ok / max(total_rollouts, 1),
    }

    return filtered_sft_dataset, stats


from drgrpo_grader import r1_zero_reward_fn
from SFT import algorithm1_sft

gen_cfg = dict(
    max_prompt_length=1024,
    max_new_tokens=256,
    do_sample=True,
    temperature=1.0,
    top_p=1.0,
)

def algorithm2_expert_iteration(
    model,
    tokenizer,
    optimizer,
    question_dataset: List[Dict[str, Any]],
    rollout_generate_fn,
    device,
    n_ei_steps: int,
    ei_batch_size: int,
    group_size: int,
    sft_steps_per_ei_step: int,
    sft_batch_size: int,
    gradient_accumulation_steps: int,
    gen_cfg: dict,
    rollout_batch_size: int = 8,
    reward_fn=r1_zero_reward_fn,
    max_grad_norm: float = 1.0,
):
    """
    Algorithm 2: Expert Iteration (EI)

    输入:
        model: 初始策略模型
        question_dataset: [{"question": str, "gold": str}, ...]
        rollout_generate_fn:
            接口为 rollout_generate_fn(
                model, tokenizer, prompts, group_size, device, gen_cfg, rollout_batch_size
            ) -> List[List[str]]

    输出:
        更新后的 model, 以及 EI 历史日志
    """
    ei_history = []

    for ei_step in range(1, n_ei_steps + 1):
        print(f"\n========== EI step {ei_step}/{n_ei_steps} ==========")

        # 1) 采样问题 batch Db
        question_batch = sample_question_batch(question_dataset, ei_batch_size)

        # 2) rollout 阶段
        prompts = [build_prompt(x["question"]) for x in question_batch]
        rollout_outputs_per_question = rollout_generate_fn(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            group_size=group_size,
            device=device,
            gen_cfg=gen_cfg,
            rollout_batch_size=rollout_batch_size,
        )

        # 3) reward / filter
        filtered_sft_dataset, reward_stats = build_sft_dataset_from_rollouts(
            question_batch,
            rollout_outputs_per_question,
            reward_fn=reward_fn,
        )

        print(
            f"[EI step {ei_step}] total_rollouts={reward_stats['total_rollouts']}, "
            f"format_ok={reward_stats['n_format_ok']}, "
            f"correct={reward_stats['n_correct']}, "
            f"keep_ratio={reward_stats['keep_ratio']:.4f}"
        )

        # 4) 如果没有正确样本，就跳过本轮 SFT
        if len(filtered_sft_dataset) == 0:
            step_metrics = {
                "ei_step": ei_step,
                **reward_stats,
                "sft_skipped": 1,
            }
            ei_history.append(step_metrics)
            print(f"[EI step {ei_step}] 没有过滤出正确轨迹，跳过 SFT。")
            continue

        # 5) 对过滤出的正确样本做 SFT
        model, sft_history = algorithm1_sft(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            sft_dataset=filtered_sft_dataset,
            device=device,
            n_sft_steps=sft_steps_per_ei_step,
            batch_size=sft_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
        )

        step_metrics = {
            "ei_step": ei_step,
            **reward_stats,
            "sft_skipped": 0,
            "filtered_sft_size": len(filtered_sft_dataset),
            "last_sft_loss": sft_history[-1]["loss"] if len(sft_history) > 0 else None,
            "last_sft_entropy": sft_history[-1]["avg_token_entropy"] if len(sft_history) > 0 else None,
        }
        ei_history.append(step_metrics)

    return model, ei_history

### 向前生成G个答案
def rollout_generate_fn_single_gpu(
    model,
    tokenizer,
    prompts,
    group_size,
    device,
    gen_cfg,
    rollout_batch_size=8,
):
    """
    单卡 EI rollout:
    - 不复制 old_model
    - 先完整 rollout
    - 返回每个 prompt 的 G 个 response
    """
    model.eval()

    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * group_size)

    all_responses = []

    with torch.no_grad():
        for start in range(0, len(expanded_prompts), rollout_batch_size):
            batch_prompts = expanded_prompts[start:start + rollout_batch_size]

            batch_full_texts = generate_batch(
                batch_prompts,
                tokenizer=tokenizer,
                model=model,
                device=device,
                gen_cfg=gen_cfg,
            )

            for prompt, full_text in zip(batch_prompts, batch_full_texts):
                # 如果 generate_batch 返回的是 prompt + response，就裁掉 prompt
                if full_text.startswith(prompt):
                    response = full_text[len(prompt):]
                else:
                    response = full_text
                all_responses.append(response)

    outputs_per_question = []
    idx = 0
    for _ in range(len(prompts)):
        outputs_per_question.append(all_responses[idx: idx + group_size])
        idx += group_size

    return outputs_per_question