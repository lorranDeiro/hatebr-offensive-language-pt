"""Milestone 2 CNN (Embedding -> Conv1D(128,5) -> GlobalMaxPool -> Dense(64) -> Dropout -> sigmoid).

original : tokenizer fitted on all 7,000 texts, 10 epochs, test set passed as validation_data
ablation : tokenizer fitted on train only, same 10 fixed epochs
corrected: tokenizer fitted on train only, early stopping on a validation split carved from train,
           test set touched once; 5 seeds, because a single CNN run is not a result.
corrected_drop_unknown: as corrected, but with no <OOV> token (unknown words are dropped)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hatebr.data import SEED, fixed_split, load_hatebr  # noqa: E402
from hatebr.metrics import scores  # noqa: E402
from hatebr.ml import cleaned_text  # noqa: E402

MAX_WORDS, MAXLEN = 10_000, 100


def build(seed):
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([
        layers.Input((MAXLEN,)), layers.Embedding(MAX_WORDS + 1, 100),
        layers.Conv1D(128, 5, activation="relu"), layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"), layers.Dropout(0.5), layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def seqs(tok, texts):
    return tf.keras.utils.pad_sequences(tok.texts_to_sequences(texts), maxlen=MAXLEN, padding="post", truncating="post")


def main(seeds=(0, 1, 2, 3, 4)):
    from tensorflow.keras.preprocessing.text import Tokenizer  # legacy API, as in the original

    df = load_hatebr()
    df["clean"] = cleaned_text(df["text"])
    split = fixed_split(df)
    ytr, yte = split.train["label"].values, split.test["label"].values
    out = {"original": [], "train_tokenizer_10_epochs": [], "corrected": [], "corrected_drop_unknown": []}

    tok_all = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok_all.fit_on_texts(df["clean"])
    tok_tr = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok_tr.fit_on_texts(split.train["clean"])

    # Train-only tokenizer WITHOUT an <OOV> token: unknown words are dropped instead of
    # becoming a token whose embedding learned a class during training (see README).
    tok_drop = Tokenizer(num_words=MAX_WORDS)
    tok_drop.fit_on_texts(split.train["clean"])

    for s in seeds:
        # original protocol
        m = build(s)
        m.fit(seqs(tok_all, split.train["clean"]), ytr, epochs=10, batch_size=32, verbose=0)
        p = (m.predict(seqs(tok_all, split.test["clean"]), verbose=0).ravel() > .5).astype(int)
        out["original"].append(scores(yte, p))

        # ablation: only the tokenizer changes (fitted on train), same 10 fixed epochs
        m = build(s)
        m.fit(seqs(tok_tr, split.train["clean"]), ytr, epochs=10, batch_size=32, verbose=0)
        p = (m.predict(seqs(tok_tr, split.test["clean"]), verbose=0).ravel() > .5).astype(int)
        out["train_tokenizer_10_epochs"].append(scores(yte, p))

        # corrected protocol
        Xtr = seqs(tok_tr, split.train["clean"])
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, random_state=s, stratify=ytr)
        m = build(s)
        m.fit(Xa, ya, epochs=30, batch_size=32, verbose=0, validation_data=(Xv, yv),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
        p = (m.predict(seqs(tok_tr, split.test["clean"]), verbose=0).ravel() > .5).astype(int)
        out["corrected"].append(scores(yte, p))

        # corrected + unknown words dropped
        Xtr = seqs(tok_drop, split.train["clean"])
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, random_state=s, stratify=ytr)
        m = build(s)
        m.fit(Xa, ya, epochs=30, batch_size=32, verbose=0, validation_data=(Xv, yv),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)])
        p = (m.predict(seqs(tok_drop, split.test["clean"]), verbose=0).ravel() > .5).astype(int)
        out["corrected_drop_unknown"].append(scores(yte, p))
        print(f"seed {s}: original F1={out['original'][-1]['f1']:.4f}  tok-train F1={out['train_tokenizer_10_epochs'][-1]['f1']:.4f}  corrected F1={out['corrected'][-1]['f1']:.4f}  drop-unk F1={out['corrected_drop_unknown'][-1]['f1']:.4f}")

    summary = {k: {"f1_mean": float(np.mean([r["f1"] for r in v])), "f1_std": float(np.std([r["f1"] for r in v])),
                   "acc_mean": float(np.mean([r["accuracy"] for r in v])), "runs": v} for k, v in out.items()}
    for k, v in summary.items():
        print(f"CNN {k}: F1 {v['f1_mean']:.4f} ± {v['f1_std']:.4f} (5 seeds)")
    (ROOT / "results" / "cnn.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
