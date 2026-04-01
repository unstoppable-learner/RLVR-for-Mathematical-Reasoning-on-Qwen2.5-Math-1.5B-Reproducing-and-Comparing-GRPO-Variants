import re
import json
from typing import List, Dict, Any, Optional

import torch


## r1_zero 的标准 prompt 格式 ====================================
# def build_prompt(question: str) -> str:
#     return (
#         "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n"
#         f"User: {question}\n"
#         "Assistant: <think>"
#     )

def build_prompt(question: str) -> str:
    return (
        "A conversation between User and Assistant. "
        "The User asks a question, and the Assistant solves it.\n"
        "The Assistant must respond in exactly this format:\n"
        "<think> reasoning process </think> <answer> final answer </answer>\n"
        "Do not include any calculations, intermediate steps, or explanations in the <answer> tag."
        "The <answer> tag should only contain the final answer, not the reasoning process.\n"
        f"User: {question}\n"
        "Assistant: <think>" 
    )

# def build_prompt(question: str) -> str:
#     return (
#         "A conversation between User and Assistant. "
#         "The User asks a question, and the Assistant solves it.\n"

#         "The Assistant must respond in exactly this format:\n"
#         "<think> reasoning process </think> <answer> final answer </answer>\n"

#         "STRICT RULES:\n"
#         "1. Do not output anything outside these tags.\n"
#         "2. Do not include any calculations, intermediate steps, or explanations in the <answer> tag.\n"
#         "3. The <answer> tag must contain ONLY the final result.\n"
#         "4. If the answer is a number, output ONLY the number (no words, no units, no explanation).\n"
#         "5. Do not include phrases like 'The answer is', 'Therefore', etc.\n"

#         f"User: {question}\n"
#         "Assistant: <think>"
#     )
### 从完整输出中截取 Assistant 的回答部分 ===========================
def extract_assistant_response(full_text: str) -> str:
    if "Assistant:" in full_text:
        return full_text.split("Assistant:", 1)[1].strip()
    return full_text.strip()


### 抓取模型的 answer（调试用，不再作为主评分逻辑）===================
ANS_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)

def extract_pred_answer(text: str):
    m = ANS_RE.search(text or "")
    return m.group(1).strip() if m else None


### 加载 jsonl 文件 ===============================================
def load_jsonl(path: str, limit: int | None = None):
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            if rec.get("gold") is None:
                continue
            data.append(rec)
    return data


### 批量生成（把 tokenizer/model/device 显式传入）====================
@torch.no_grad()
def generate_batch(
    prompts: List[str],
    tokenizer,
    model,
    device: torch.device,
    gen_cfg: Optional[Dict[str, Any]] = None,
) -> List[str]:
    gen_cfg = gen_cfg or {}

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=gen_cfg.get("max_prompt_length", 1024),
    ).to(device)

    out = model.generate(
        **enc,
        max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        do_sample=gen_cfg.get("do_sample", True),
        temperature=gen_cfg.get("temperature", 1.0),
        top_p=gen_cfg.get("top_p", 1.0),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
    )

    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return texts


### 简化答案（调试用，不再作为主评分逻辑）===========================
def normalize_answer(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()

    s = s.replace("\\,", "").replace("\\!", "")
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"^\\boxed\s*\{?(.*?)\}?$", r"\1", s).strip()
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = s.strip("$")
    s = re.sub(r"\s+", "", s)

    return s

import os
from pathlib import Path


def generate_batch_vllm(
    llm,
    prompts: List[str],
    sampling_params=None,
    stop: Optional[List[str]] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
) -> List[str]:
    """
    Use vLLM to generate one response for each prompt.

    Args:
        llm: vllm.LLM instance
        prompts: list of prompt strings
        sampling_params: optional prebuilt vllm.SamplingParams
        stop: optional stop strings; if sampling_params is given, this is ignored
        temperature: sampling temperature
        top_p: nucleus sampling top_p
        max_tokens: max generated tokens

    Returns:
        List[str]: generated texts, aligned with prompts
    """
    if sampling_params is None:
        try:
            from vllm import SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Please install vllm before calling generate_batch_vllm."
            ) from e

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop if stop is not None else ["</answer>"],
            include_stop_str_in_output=True,
            n=1,
        )

    outputs = llm.generate(prompts, sampling_params)

    generations = []
    for out in outputs:
        if not out.outputs:
            generations.append("")
        else:
            generations.append(out.outputs[0].text)

    return generations


def save_results_json(
    results: List[Dict[str, Any]],
    save_path: str,
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save evaluation results to disk.

    If save_path endswith '.jsonl', write one record per line.
    Otherwise write a json dict with keys: summary, results.

    Args:
        results: per-example records
        save_path: output path
        summary: optional summary metrics
    """
    save_path = str(save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if save_path.endswith(".jsonl"):
        with open(save_path, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        payload = {
            "summary": summary or {},
            "results": results,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def evaluate_vllm(
    llm,
    dataset: List[Dict[str, Any]],
    reward_fn,
    batch_size: int = 64,
    sampling_params=None,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model with vLLM on a dataset.

    Expected dataset item format:
        {
            "question": str,
            "gold": str | list[str] | number,
            ...
        }

    Returns:
        {
            "summary": {...},
            "results": [...]
        }
    """
    results: List[Dict[str, Any]] = []

    n_total = 0
    n_format_and_correct = 0
    n_format_and_wrong = 0
    n_unformatted = 0

    sum_reward = 0.0
    sum_answer_reward = 0.0
    sum_format_reward = 0.0

    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]

        prompts = [build_prompt(ex["question"]) for ex in batch]
        completions = generate_batch_vllm(
            llm=llm,
            prompts=prompts,
            sampling_params=sampling_params,
        )

        for ex, prompt, completion in zip(batch, prompts, completions):
            response = completion.strip()
            reward_out = reward_fn(response, ex["gold"])

            format_reward = float(reward_out.get("format_reward", 0.0))
            answer_reward = float(reward_out.get("answer_reward", 0.0))
            reward = float(reward_out.get("reward", 0.0))

            n_total += 1
            sum_format_reward += format_reward
            sum_answer_reward += answer_reward
            sum_reward += reward

            if format_reward == 1.0 and answer_reward == 1.0:
                n_format_and_correct += 1
            elif format_reward == 1.0 and answer_reward == 0.0:
                n_format_and_wrong += 1
            elif format_reward == 0.0 and answer_reward == 0.0:
                n_unformatted += 1

            results.append(
                {
                    "id": ex.get("id"),
                    "question": ex["question"],
                    "gold": ex["gold"],
                    "level": ex.get("level"),
                    "type": ex.get("type"),
                    "solution": ex.get("solution"),
                    "prompt": prompt,
                    "response": response,
                    "format_reward": format_reward,
                    "answer_reward": answer_reward,
                    "reward": reward,
                    "pred_answer_debug": extract_pred_answer(response),
                }
            )

        if verbose:
            done = min(start + batch_size, len(dataset))
            acc_so_far = sum_answer_reward / max(n_total, 1)
            print(
                f"[{done}/{len(dataset)}] "
                f"acc={acc_so_far:.4f}, "
                f"formatted_correct={n_format_and_correct}, "
                f"formatted_wrong={n_format_and_wrong}, "
                f"unformatted={n_unformatted}"
            )

    summary = {
        "num_examples": n_total,
        "accuracy": sum_answer_reward / max(n_total, 1),
        "avg_answer_reward": sum_answer_reward / max(n_total, 1),
        "avg_format_reward": sum_format_reward / max(n_total, 1),
        "avg_reward": sum_reward / max(n_total, 1),
        "formatted_and_correct": n_format_and_correct,
        "formatted_and_wrong": n_format_and_wrong,
        "unformatted": n_unformatted,
    }

    if save_path is not None:
        save_results_json(results=results, save_path=save_path, summary=summary)

    return {
        "summary": summary,
        "results": results,
    }


def main():
    """
    Example zero-shot baseline entrypoint.

    Default assumptions:
    - model path: /data/a5-alignment/models/Qwen2.5-Math-1.5B
    - val set:    /data/a5-alignment/MATH/validation.jsonl
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ImportError(
            "vLLM is not installed. Please install vllm first."
        ) from e

    from drgrpo_grader import r1_zero_reward_fn

    model_path = "/home/u2022310886/jupyterlab/RL/models/Qwen2.5-Math-1.5B"
    val_path = os.environ.get(
        "CS336_VAL_PATH",
        "/data/a5-alignment/MATH/validation.jsonl",
    )
    out_path = os.environ.get(
        "CS336_OUT_PATH",
        "baseline_results/baseline_val_results.json",
    )
    batch_size = int(os.environ.get("CS336_BATCH_SIZE", 64))
    limit_env = os.environ.get("CS336_LIMIT", None)
    limit = int(limit_env) if limit_env is not None and limit_env != "" else None

    print(f"Loading validation set from: {val_path}")
    dataset = load_jsonl(val_path, limit=limit)
    print(f"Loaded {len(dataset)} examples")

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=1,
    )

    print(f"Loading model from: {model_path}")
    llm = LLM(model=model_path,
             trust_remote_code=True,
             )

    eval_out = evaluate_vllm(
        llm=llm,
        dataset=dataset,
        reward_fn=r1_zero_reward_fn,
        batch_size=batch_size,
        sampling_params=sampling_params,
        save_path=out_path,
        verbose=True,
    )

    print("\n=== Final Summary ===")
    for k, v in eval_out["summary"].items():
        print(f"{k}: {v}")

    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()