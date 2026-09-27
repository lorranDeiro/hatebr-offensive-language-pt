"""Milestone 1, original protocol vs. corrected protocol.

original : lexicon built from all 7,000 comments, scored on the same 7,000
corrected: lexicon built from the 5,250 training comments, scored on the 1,750 test comments
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatebr.data import fixed_split, load_hatebr  # noqa: E402
from hatebr.metrics import report  # noqa: E402
from hatebr.rules import RuleClassifier, build_lexicon  # noqa: E402


def main():
    df = load_hatebr()
    split = fixed_split(df)
    out = {}

    lex_all = build_lexicon(df)
    clf = RuleClassifier(lex_all)
    out["rules_original_protocol_all7000"] = report(
        "Rules | lexicon=all, eval=all 7000 (original)", df["label"], clf.predict(df["text"]))
    out["rules_original_protocol_on_test"] = report(
        "Rules | lexicon=all, eval=test (test seen)", split.test["label"], clf.predict(split.test["text"]))

    lex_train = build_lexicon(split.train)
    clf = RuleClassifier(lex_train)
    out["rules_corrected"] = report(
        "Rules | lexicon=train, eval=test (corrected)", split.test["label"], clf.predict(split.test["text"]))
    out["lexicon_sizes"] = {
        "all": {k: len(v) for k, v in lex_all.items()},
        "train": {k: len(v) for k, v in lex_train.items()},
    }
    # Share of the "offensive" words learned from all data that the train-only lexicon never learns
    only_via_test = lex_all["offensive"] - lex_train["offensive"]
    out["lexicon_sizes"]["offensive_words_only_learnable_with_test"] = len(only_via_test)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rules.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=list))
    print(json.dumps(out["lexicon_sizes"], indent=2))


if __name__ == "__main__":
    main()
