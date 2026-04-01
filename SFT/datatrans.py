import json
from pathlib import Path


def convert_jsonl_with_response(
    input_path="/home/u2022310886/jupyterlab/RL/math_lighteval_train.jsonl",
    output_path="/home/u2022310886/jupyterlab/RL/math_lighteval_train_fitted.jsonl",
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    n_total = 0
    n_written = 0
    n_skipped = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[跳过] 第 {n_total} 行不是合法 JSON")
                n_skipped += 1
                continue

            solution = str(obj.get("solution", "")).strip()
            gold = str(obj.get("gold", "")).strip()

            response = f'{solution}</think> <answer>{gold}</answer>'

            new_obj = {
                "id": obj.get("id", ""),
                "question": obj.get("question", ""),
                "gold": obj.get("gold", ""),
                "level": obj.get("level", ""),
                "type": obj.get("type", ""),
                "response": response,
            }

            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"处理完成")
    print(f"总行数: {n_total}")
    print(f"成功写入: {n_written}")
    print(f"跳过条数: {n_skipped}")
    print(f"输出文件: {output_path}")


# 直接执行
convert_jsonl_with_response()