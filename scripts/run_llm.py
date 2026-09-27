"""Run a prompt variant with a local Ollama model.

    ollama pull llama3.2:3b
    # 1) choose prompts on a 300-comment validation sample taken from the TRAINING split
    python scripts/run_llm.py --strategy zs-original --on val
    python scripts/run_llm.py --strategy fs-original --on val
    python scripts/run_llm.py --strategy cot-512     --on val
    # 2) run only the chosen prompt per family on the 1,750 test comments
    python scripts/run_llm.py --strategy zs-original --on test

Variants are listed in src/hatebr/llm.py (STRATEGIES). Raw answers are cached in
results/llm_raw_<split>_<strategy>_<model>.jsonl, so an interrupted run resumes;
scripts/score_llm.py re-scores every cache without calling the model.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hatebr.data import fixed_split, load_hatebr  # noqa: E402
from hatebr.llm import MODEL, STRATEGIES, generate, pick_few_shot, validation_sample  # noqa: E402
from score_llm import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    ap.add_argument("--on", choices=["val", "test"], default="val")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    split = fixed_split(load_hatebr())
    data = validation_sample(split.train) if args.on == "val" else split.test
    shots = pick_few_shot(split.train)
    st = STRATEGIES[args.strategy]

    cache_path = ROOT / "results" / f"llm_raw_{args.on}_{args.strategy}_{args.model.replace(':', '-')}.jsonl"
    done = set()
    if cache_path.exists():
        done = {json.loads(line)["id"] for line in cache_path.read_text(encoding="utf-8").splitlines()}

    with cache_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(data.itertuples(), 1):
            if row.id in done:
                continue
            t0 = time.time()
            ans = generate(st.build(row.text, shots), st.num_predict, model=args.model)
            f.write(json.dumps({"id": int(row.id), "label": int(row.label), "answer": ans,
                                "seconds": time.time() - t0}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(data)}")

    res = score(cache_path)
    cache_path.with_name(cache_path.name.replace("llm_raw_", "llm_").replace(".jsonl", ".json")).write_text(
        json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
