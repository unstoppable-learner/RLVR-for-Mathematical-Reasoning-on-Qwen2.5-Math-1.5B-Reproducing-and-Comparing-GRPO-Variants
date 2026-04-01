import json
from datasets import load_dataset
import re

BOXED_BARE_RE = re.compile(r"\\boxed\s+([^\s\.\,;\!\?]+)")

def extract_boxed_answer(solution: str):
    """
    Supports:
    1) \\boxed{...} with nested braces
    2) \\boxed <token> (no braces), e.g. \\boxed 2
    """
    if not solution:
        return None

    # ---- 1) Try \\boxed{...} with nested braces ----
    key = r"\boxed{"
    start = solution.find(key)
    if start != -1:
        i = start + len(key)
        depth = 1
        out = []
        while i < len(solution):
            ch = solution[i]
            if ch == "{":
                depth += 1
                out.append(ch)
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(out).strip()
                out.append(ch)
            else:
                out.append(ch)
            i += 1
        # unbalanced
        return None

    # ---- 2) Fallback: \\boxed 2 (no braces) ----
    m = BOXED_BARE_RE.search(solution)
    if m:
        return m.group(1).strip()

    return None

def write_split(ds, split: str, out_path: str, limit=None):
    rows = ds[split]
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))

    n_total = len(rows)
    n_gold = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(rows):
            gold = extract_boxed_answer(ex.get("solution", ""))
            if gold is not None:
                n_gold += 1

            rec = {
                "id": f"{split}-{i:06d}",
                "question": ex.get("problem"),
                "gold": gold,
                "level": ex.get("level"),
                "type": ex.get("type"),
                "solution": ex.get("solution"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] wrote {out_path}")
    print(f"[stats:{split}] total={n_total}, gold_extracted={n_gold}, rate={n_gold/n_total:.3f}")

def main():
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")

    limit = None  # 调试：比如 20；正式：None
    write_split(ds, "train", "math_lighteval_train.jsonl", limit=limit)
    write_split(ds, "test",  "math_lighteval_test.jsonl",  limit=limit)

if __name__ == "__main__":
    main()