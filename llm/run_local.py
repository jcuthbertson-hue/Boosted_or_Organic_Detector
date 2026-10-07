"""Run the harness prompts through any OpenAI-compatible endpoint (LM Studio, Ollama, llama.cpp server, vLLM)
and score the answers.

Example (LM Studio on your Mac, model loaded as "decider-4b"):
  python3 -m llm.run_local --base-url http://localhost:1234/v1 --model decider-4b
Example (Ollama):
  python3 -m llm.run_local --base-url http://localhost:11434/v1 --model decider:4b

Writes llm/harness/answers_<model>.csv and prints ROC-AUC, PR-AUC, precision, recall, F1 by platform,
next to the GPT-5 / Claude / Llama / ML numbers in results/llm_vs_ml.csv.
"""
import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

from llm.score_llm import parse, score


def ask(base_url, model, prompt, timeout=120):
    body = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open("llm/harness/test_prompts.jsonl")]
    out = []
    for i, r in enumerate(rows, 1):
        ans = ask(a.base_url, a.model, r["prompt"])
        out.append({"row_id": r["row_id"], "platform": r["platform"], "answer": ans.replace("\n", " ")})
        if i % 25 == 0:
            print(f"{i}/{len(rows)}")
    ans = pd.DataFrame(out)
    safe = a.model.replace("/", "_").replace(":", "_")
    ans.to_csv(f"llm/harness/answers_{safe}.csv", index=False)
    key = pd.read_csv("llm/harness/answer_key.csv")
    j = key.merge(ans, on=["row_id", "platform"])
    j["p"] = j.answer.map(parse)
    j["y"] = (j.label == "POS").astype(int)
    for plat, g in j.groupby("platform"):
        ok = g.p.notna()
        print(plat, f"parsed {ok.sum()}/{len(g)}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in score(g.y[ok], g.p[ok]).items()})


if __name__ == "__main__":
    main()
