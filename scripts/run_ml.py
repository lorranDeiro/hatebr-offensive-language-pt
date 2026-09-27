"""Milestone 2: original models (TF-IDF fitted before the split) vs. Pipelines fitted on train only,
plus a word+char n-gram SVM tuned with 5-fold CV on the training split, and a grouped check
(no Instagram post shared between train and test)."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatebr.data import SEED, fixed_split, load_hatebr  # noqa: E402
from hatebr.metrics import report  # noqa: E402
from hatebr.ml import cleaned_text, original_models, word_char_svm  # noqa: E402


def main():
    df = load_hatebr()
    df["clean"] = cleaned_text(df["text"])
    split = fixed_split(df)
    out, preds = {}, {}

    # 1) Original protocol: TF-IDF fitted on all 7,000 texts, then split
    X_all = TfidfVectorizer(max_features=5000).fit_transform(df["clean"])
    Xtr, Xte, ytr, yte = train_test_split(X_all, df["label"], test_size=0.25, random_state=SEED, stratify=df["label"])
    for name, pipe in original_models().items():
        clf = clone(pipe.named_steps["clf"]).fit(Xtr, ytr)
        out[f"{name} (original: TF-IDF fit on all)"] = report(f"{name} | TF-IDF fit on all", yte, clf.predict(Xte))

    # 2) Same models, TF-IDF fitted on train only
    for name, pipe in original_models().items():
        p = pipe.fit(split.train["clean"], split.train["label"]).predict(split.test["clean"])
        preds[name] = p.tolist()
        out[f"{name} (corrected)"] = report(f"{name} | Pipeline, train only", split.test["label"], p)

    # 3) Word+char SVM, C chosen by 5-fold CV on train only (raw text, lowercase)
    gs = GridSearchCV(word_char_svm(), {"clf__C": [0.1, 0.25, 0.5, 1.0]}, scoring="f1",
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=2)
    gs.fit(split.train["text"], split.train["label"])
    p = gs.predict(split.test["text"])
    preds["Word+char SVM"] = p.tolist()
    out["Word+char SVM (CV-tuned)"] = report(f"Word+char SVM | C={gs.best_params_['clf__C']}", split.test["label"], p)
    out["Word+char SVM (CV-tuned)"]["best_C"] = gs.best_params_["clf__C"]
    out["Word+char SVM (CV-tuned)"]["cv_f1_train"] = gs.best_score_

    # 4) Grouped robustness check: 5 folds, no Instagram post in both train and test
    gkf = GroupKFold(n_splits=5)
    fold_f1 = {"Naive Bayes": [], "Word+char SVM": []}
    rnd_f1 = {"Naive Bayes": [], "Word+char SVM": []}
    for tr, te in gkf.split(df, df["label"], groups=df["links_post"]):
        a, b = df.iloc[tr], df.iloc[te]
        fold_f1["Naive Bayes"].append(f1_score(b["label"], original_models()["Naive Bayes"].fit(a["clean"], a["label"]).predict(b["clean"])))
        fold_f1["Word+char SVM"].append(f1_score(b["label"], word_char_svm(gs.best_params_["clf__C"]).fit(a["text"], a["label"]).predict(b["text"])))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, df["label"]):
        a, b = df.iloc[tr], df.iloc[te]
        rnd_f1["Naive Bayes"].append(f1_score(b["label"], original_models()["Naive Bayes"].fit(a["clean"], a["label"]).predict(b["clean"])))
        rnd_f1["Word+char SVM"].append(f1_score(b["label"], word_char_svm(gs.best_params_["clf__C"]).fit(a["text"], a["label"]).predict(b["text"])))
    out["group_vs_random_5fold_f1"] = {
        k: {"random_mean": float(np.mean(rnd_f1[k])), "random_std": float(np.std(rnd_f1[k])),
            "grouped_by_post_mean": float(np.mean(fold_f1[k])), "grouped_by_post_std": float(np.std(fold_f1[k]))}
        for k in fold_f1
    }
    print(json.dumps(out["group_vs_random_5fold_f1"], indent=2))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ml.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (ROOT / "results" / "ml_test_predictions.json").write_text(json.dumps(preds))


if __name__ == "__main__":
    main()
