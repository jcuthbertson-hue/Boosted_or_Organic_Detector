"""Export the 300 locked-test LLM prompts so any local model (for example Decider 2B / 4B) can be scored
on exactly the same posts as GPT-5, Claude and Llama.

Writes (no creator names, no client names, no platform post IDs):
  llm/harness/test_prompts.jsonl   {"row_id", "platform", "prompt"}
  llm/harness/answer_key.csv       row_id, platform, label  (POS = boosted, NEG = organic)

Run from repo root:  python3 -m llm.export_prompts
"""
import json
import os

import pandas as pd

from detector.features import load

TEMPLATE = open(os.path.join(os.path.dirname(__file__), "prompt.txt")).read()


def fmt(x):
    return "unknown" if pd.isna(x) else str(int(round(float(x))))


def main():
    ids = pd.read_csv("results/llm/openai-gpt-5.psv", sep="|", header=None, dtype=str, engine="python", quoting=3)[[0, 1]]
    ids.columns = ["psrk", "platform"]
    d = load().merge(ids, on=["psrk", "platform"])
    d = d.sort_values(["platform", "psrk"]).reset_index(drop=True)
    d["row_id"] = [f"T{i:04d}" for i in range(len(d))]
    os.makedirs("llm/harness", exist_ok=True)
    with open("llm/harness/test_prompts.jsonl", "w") as f:
        for _, r in d.iterrows():
            prompt = TEMPLATE.format(platform=r.platform, followers=fmt(r.followers), v1=fmt(r.v1), v3=fmt(r.v3), v7=fmt(r.v7),
                                     v14=fmt(r.v14), v30=fmt(r.v30), l7=fmt(r.l7), l30=fmt(r.l30), c30=fmt(r.c30), s30=fmt(r.s30)).strip()
            f.write(json.dumps({"row_id": r.row_id, "platform": r.platform, "prompt": prompt}) + "\n")
    d[["row_id", "platform", "label"]].to_csv("llm/harness/answer_key.csv", index=False)
    # private mapping row_id -> psrk stays out of git (data/ is ignored)
    d[["row_id", "psrk", "platform"]].to_csv("data/llm_harness_row_map.csv", index=False)
    print(len(d), "prompts written")


if __name__ == "__main__":
    main()
