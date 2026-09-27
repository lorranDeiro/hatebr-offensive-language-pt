"""Score cached LLM answers (results/llm_raw_<split>_<strategy>_<model>.jsonl) without calling the model.

Two numbers per run:
  strict      : abstentions (refusals / no label) count as errors -> headline number
  refusal_is_1: a refusal is read as "offensive"                   -> sensitivity check
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatebr.llm import _REFUSAL, STRATEGIES, parse  # noqa: E402
from hatebr.metrics import report  # noqa: E402

NAME = re.compile(r"llm_raw_(val|test)_(.+)_(.+)\.jsonl$")


def score(raw_path: Path) -> dict:
    split, strategy, model = NAME.search(raw_path.name).groups()
    mode = STRATEGIES[strategy].parse_mode
    recs = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    y = [r["label"] for r in recs]
    p = [parse(r["answer"], mode) for r in recs]
    refused = [v is None and bool(_REFUSAL.search(r["answer"])) for v, r in zip(p, recs)]
    tag = f"{split:<4} {strategy}"
    res = report(f"{tag} (strict)", y, [v if v is not None else 1 - t for v, t in zip(p, y)])
    alt = report(f"{tag} (refusal=1)", y, [v if v is not None else (1 if rf else 1 - t)
                                          for v, t, rf in zip(p, y, refused)])
    res.update({
        "split": split, "strategy": strategy, "model": model,
        "abstentions": sum(v is None for v in p),
        "refusals": sum(refused),
        "refusals_on_offensive": sum(rf and t == 1 for rf, t in zip(refused, y)),
        "predicted_offensive_rate": sum(v == 1 for v in p) / len(p),
        "f1_if_refusal_means_offensive": alt["f1"],
        "mean_seconds": sum(r["seconds"] for r in recs) / len(recs),
    })
    print(f"      abstentions={res['abstentions']} (refusals={res['refusals']}, {res['refusals_on_offensive']} on "
          f"offensive)  predicted offensive={res['predicted_offensive_rate']:.1%}  {res['mean_seconds']:.1f}s/text")
    return res


def main():
    for raw in sorted((ROOT / "results").glob("llm_raw_*.jsonl")):
        if not NAME.search(raw.name):
            continue
        res = score(raw)
        raw.with_name(raw.name.replace("llm_raw_", "llm_").replace(".jsonl", ".json")).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
