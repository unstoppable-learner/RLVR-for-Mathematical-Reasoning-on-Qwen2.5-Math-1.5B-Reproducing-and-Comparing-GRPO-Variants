import os
import re
import json
import argparse
from tqdm import tqdm  # 引入 tqdm 用于进度条显示
from transformers import AutoModelForCausalLM, AutoTokenizer
from baseline_funs import generate_batch_vllm, load_jsonl
from drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams


def run_math_baseline(limit=None):
    """
    运行基准测试，可以选择只测试前 n 个样本，或者全部样本。

    参数:
        limit (int, optional): 如果指定，测试前 n 个样本。如果为 None，则测试所有样本。
    """
    # 配置文件路径
    data_path = "/home/u2022310886/jupyterlab/RL/math_lighteval_test.jsonl"
    model_path = "/home/u2022310886/jupyterlab/RL/models/Qwen2.5-Math-1.5B-SFT-GRPO-lambda-epsTrace-0.5beta"  # 替换为正确的模型路径
    result_path = "/home/u2022310886/jupyterlab/RL/GRPO/result_grpo_lambda_noadvclip_0.5beta/grpo_lambda_0.5beta_result.json"

    # 加载数据集
    print(f"Loading data from {data_path}")
    dataset = load_jsonl(data_path, limit=limit)  # 加载前n个样本或所有样本

    # 加载模型
    print(f"Loading model from {model_path}")
    try:
        llm = LLM(
            model=model_path,
            max_num_seqs=32,   # ⭐ 从256降下来（关键）
        )
    except ImportError as e:
        raise ImportError("vLLM is not installed. Please install vllm first.") from e

    # 配置生成参数
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=1,
    )

    # 生成模型的输出并评估
    print(f"Evaluating model on dataset...")
    eval_out = evaluate_vllm(
        llm=llm,
        dataset=dataset,
        reward_fn=r1_zero_reward_fn,
        batch_size=8,
        sampling_params=sampling_params,
        save_path=result_path,  # 保存结果到指定路径
        verbose=True,
    )

    # 打印最终的评估结果
    print("\n=== Final Summary ===")
    for k, v in eval_out["summary"].items():
        print(f"{k}: {v}")

    # 保存结果
    print(f"Results saved to {result_path}")

from baseline_funs import build_prompt
def evaluate_vllm(
    llm,
    dataset,
    reward_fn,
    batch_size=8,
    sampling_params=None,
    save_path=None,
    verbose=True,
):
    results = []
    n_total = 0
    n_format_and_correct = 0
    n_format_and_wrong = 0
    n_unformatted = 0

    sum_reward = 0.0
    sum_answer_reward = 0.0
    sum_format_reward = 0.0

    # 使用 tqdm 来显示进度条
    for start in tqdm(range(0, len(dataset), batch_size), desc="Processing Batches"):
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

def save_results_json(results, save_path, summary):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)


def extract_pred_answer(response):
    # 自定义从response中提取答案的函数
    m = re.search(r"<answer>(.*?)</answer>", response)
    return m.group(1).strip() if m else None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Run Math Baseline Evaluation.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of samples to test."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 传递 limit 参数来限制测试样本数量
    run_math_baseline(limit=args.limit)