"""F1 on the held-out test set: as originally reported vs. with the corrected protocol.

    python scripts/make_figure.py en   # README figure
    python scripts/make_figure.py pt   # Portuguese version
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
T = {
    "en": dict(rows=["Rules (Milestone 1)", "Naive Bayes (M2)", "Linear SVM (M2)",
                     "Logistic regression (M2)", "Word+char n-gram SVM (new)",
                     "Llama 3.2 3B zero-shot (M3)", "Llama 3.2 3B chain-of-thought (M3)"],
               x="F1 (offensive class) on the 1,750-comment test set", rep="Original protocol (reproduced; LLM: as reported)",
               cor="Corrected protocol (bar = 95% bootstrap CI)",
               title="The rule-based classifier only looked like the winner",
               out="f1_reported_vs_corrected.png"),
    "pt": dict(rows=["Regras (Meta 1)", "Naive Bayes (Meta 2)", "SVM linear (Meta 2)",
                     "Regressão logística (Meta 2)", "SVM palavra+caracteres (novo)",
                     "Llama 3.2 3B zero-shot (Meta 3)", "Llama 3.2 3B chain-of-thought (Meta 3)"],
               x="F1 (classe ofensiva), teste com 1.750 comentários", rep="Protocolo original (reproduzido; LLM: reportado)",
               cor="Protocolo corrigido (barra = IC 95% bootstrap)",
               title="O classificador de regras só parecia o melhor",
               out="f1_reportado_vs_corrigido_pt.png"),
}[LANG]

rules = json.loads((ROOT / "results/rules.json").read_text())
ml = json.loads((ROOT / "results/ml.json").read_text())
llm = {k: json.loads((ROOT / f"results/llm_test_{k}_llama3.2-3b.json").read_text()) for k in ("zs-original", "cot-512")}
REPORTED_M3 = {"zs-original": 0.7453, "cot-512": 0.7768}  # Relatorio_Meta3.pdf, table 4.2
rows = [  # label, reported F1, corrected result
    (T["rows"][0], rules["rules_original_protocol_all7000"]["f1"], rules["rules_corrected"]),
    (T["rows"][1], ml["Naive Bayes (original: TF-IDF fit on all)"]["f1"], ml["Naive Bayes (corrected)"]),
    (T["rows"][2], ml["Linear SVM (original: TF-IDF fit on all)"]["f1"], ml["Linear SVM (corrected)"]),
    (T["rows"][3], ml["Logistic Regression (original: TF-IDF fit on all)"]["f1"], ml["Logistic Regression (corrected)"]),
    (T["rows"][4], None, ml["Word+char SVM (CV-tuned)"]),
    (T["rows"][5], REPORTED_M3["zs-original"], llm["zs-original"]),
    (T["rows"][6], REPORTED_M3["cot-512"], llm["cot-512"]),
]

BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
fig, ax = plt.subplots(figsize=(8.5, 5.4), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for i, (name, rep, cor) in enumerate(rows[::-1]):
    lo, hi = cor["f1_ci95"]
    ax.plot([lo, hi], [i, i], color=BLUE, lw=2, alpha=.3, solid_capstyle="round", zorder=1)
    if rep is not None and abs(rep - cor["f1"]) > .0015:
        ax.plot([rep, cor["f1"]], [i, i], color="#b5b4ad", lw=2, zorder=2)
        ax.scatter(rep, i, s=70, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=2)
        ax.text(rep, i + .24, f"{rep:.3f}", color=MUTED, fontsize=8, va="bottom", ha="center")
    ax.scatter(cor["f1"], i, s=70, color=BLUE, zorder=4, edgecolor=SURF, linewidth=2)
    ax.text(cor["f1"], i - .26, f"{cor['f1']:.3f}", color=INK, fontsize=8, va="top", ha="center")

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]], color=INK, fontsize=9)
ax.set_xlim(.70, .90)
ax.set_ylim(-.75, len(rows) - .45)
ax.set_xlabel(T["x"], color=MUTED, fontsize=8.5)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors=MUTED, labelsize=8, length=0)
ax.grid(axis="x", color="#e6e5e0", lw=.8)
ax.set_axisbelow(True)
ax.scatter([], [], s=60, color=ORANGE, label=T["rep"])
ax.scatter([], [], s=60, color=BLUE, label=T["cor"])
ax.legend(loc="lower left", bbox_to_anchor=(-.01, 1.0), ncol=2, frameon=False, fontsize=8,
          labelcolor=INK, handletextpad=.3, columnspacing=1.4)
ax.set_title(T["title"], loc="left", pad=24, color=INK, fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "results/figures" / T["out"], facecolor=SURF)
