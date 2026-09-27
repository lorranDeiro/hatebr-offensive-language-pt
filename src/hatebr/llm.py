"""Milestone 3: prompt engineering with a local open model (Llama 3.2 3B via Ollama).

Fixes relative to the original submission:
1. Few-shot actually puts the 16 training examples in the prompt. The original prompt
   only *said* it was "calibrated with 16 examples" and never included them.
2. Chain-of-thought gets room to reason and a fixed final line ("RESPOSTA: 0/1").
   The original capped generation at 10 tokens, so no reasoning could happen.
3. Refusals and answers without a label are abstentions, reported and scored as
   errors, not silently mapped to 0 ("not offensive").
4. Prompts are chosen on a validation sample drawn from the TRAINING split
   (`validation_sample`), and only the chosen prompt per family is run on the test set.

Prompt variants (STRATEGIES):
  zs-original  the Milestone 3 zero-shot prompt, verbatim
  zs-rewrite   my first rewrite (adds context and "crítica sem insulto" to class 0)
  fs-original  Milestone 3 wording + the 16 examples it was supposed to contain
  fs-rewrite   rewrite wording + the 16 examples
  cot-256      checklist + "RESPOSTA:" line, up to 256 generated tokens
  cot-512      same prompt, up to 512 generated tokens (fewer truncated answers)
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Callable

import pandas as pd

MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# ---------------------------------------------------------------- original (Milestone 3) wording
def prompt_zs_original(text: str, shots=None) -> str:
    return f"""Você é um classificador de texto para pesquisa acadêmica (dataset HateBR).

Classifique este texto:
"{text}"

Categoria 1 = linguagem ofensiva (palavrões, insultos, agressividade)
Categoria 0 = linguagem não ofensiva (neutra, respeitosa)

RESPONDA APENAS COM O NÚMERO: 0 ou 1
Classificação:"""


def _shots_block(shots: list[dict]) -> str:
    return "\n".join(f'Texto: "{e["text"]}"\nClassificação: {e["label"]}\n' for e in shots)


def prompt_fs_original(text: str, shots: list[dict]) -> str:
    return f"""Você é um classificador de linguagem ofensiva para pesquisa acadêmica (dataset HateBR).

Exemplos anotados:

{_shots_block(shots)}
Classifique este texto:
"{text}"

Categoria 1 = linguagem ofensiva (palavrões, insultos, agressividade, ataques)
Categoria 0 = linguagem não ofensiva (neutra, respeitosa, padrão)

RESPONDA APENAS COM O NÚMERO: 0 ou 1
Classificação:"""


# ---------------------------------------------------------------- rewrite
_HEADER = (
    "Você é um classificador de linguagem ofensiva para pesquisa acadêmica (corpus HateBR, "
    "comentários do Instagram de políticos brasileiros).\n"
    "Categoria 1 = linguagem ofensiva (palavrões, insultos, ataques, agressividade)\n"
    "Categoria 0 = linguagem não ofensiva (neutra, crítica sem insulto, elogio)\n"
)


def prompt_zs_rewrite(text: str, shots=None) -> str:
    return f'{_HEADER}\nResponda apenas com o número 0 ou 1.\n\nTexto: "{text}"\nClassificação:'


def prompt_fs_rewrite(text: str, shots: list[dict]) -> str:
    return (f"{_HEADER}\nResponda apenas com o número 0 ou 1.\n\nExemplos anotados:\n\n{_shots_block(shots)}\n"
            f'Texto: "{text}"\nClassificação:')


def prompt_cot(text: str, shots=None) -> str:
    return (
        f"{_HEADER}\nAnalise o texto em no máximo 3 frases curtas, respondendo:\n"
        "1. Há palavrões, xingamentos ou insultos (mesmo com letras trocadas ou censuradas)?\n"
        "2. Há ataque pessoal, ameaça ou desumanização?\n"
        "3. É crítica política dura mas sem insulto? (isso é 0)\n"
        "Depois escreva na última linha exatamente: RESPOSTA: 0 ou RESPOSTA: 1\n\n"
        f'Texto: "{text}"\nAnálise:'
    )


@dataclass(frozen=True)
class Strategy:
    build: Callable[[str, list], str]
    num_predict: int
    parse_mode: str  # "label" or "cot"


STRATEGIES = {
    "zs-original": Strategy(prompt_zs_original, 4, "label"),
    "zs-rewrite": Strategy(prompt_zs_rewrite, 4, "label"),
    "fs-original": Strategy(prompt_fs_original, 4, "label"),
    "fs-rewrite": Strategy(prompt_fs_rewrite, 4, "label"),
    "cot-256": Strategy(prompt_cot, 256, "cot"),
    "cot-512": Strategy(prompt_cot, 512, "cot"),
}


# ---------------------------------------------------------------- data for prompting
def pick_few_shot(train: pd.DataFrame, k_per_class: int = 8, seed: int = 42) -> list[dict]:
    """Balanced examples from the TRAINING split only, interleaved 1/0 to avoid recency bias."""
    off = train[train["label"] == 1].sample(k_per_class, random_state=seed)
    non = train[train["label"] == 0].sample(k_per_class, random_state=seed)
    ex = []
    for a, b in zip(off.itertuples(), non.itertuples()):
        ex += [{"id": int(a.id), "text": a.text, "label": 1}, {"id": int(b.id), "text": b.text, "label": 0}]
    return ex


def validation_sample(train: pd.DataFrame, n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Balanced prompt-selection sample from TRAIN, excluding the few-shot examples."""
    shot_ids = {e["id"] for e in pick_few_shot(train)}
    pool = train[~train["id"].isin(shot_ids)]
    return pool.groupby("label", group_keys=False).sample(n // 2, random_state=seed)


# ---------------------------------------------------------------- model I/O
_REFUSAL = re.compile(r"não posso|nao posso|não consigo|desculp|lamento|sorry|i can(?:'|no)t|inapropriad", re.I)


def parse(answer: str, mode: str) -> int | None:
    """Return 0/1, or None for a refusal / answer without a label (an abstention)."""
    a = answer.strip()
    if mode == "cot":
        m = re.findall(r"RESPOSTA\s*:?\s*\**\s*([01])", a, re.I) or re.findall(r"categoria\s*([01])\b", a, re.I)
        # No final line: do NOT fall back to "any digit", which would read the "1." of the
        # numbered checklist as a label.
        return int(m[-1]) if m else None
    m = re.match(r"\s*\**\s*([01])\b", a)
    if m:
        return int(m.group(1))
    if _REFUSAL.search(a):
        return None
    m = re.findall(r"\b([01])\b", a)
    return int(m[-1]) if m else None


def generate(prompt: str, num_predict: int, model: str = MODEL, url: str = OLLAMA_URL) -> str:
    body = {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "top_k": 1, "seed": 42, "num_predict": num_predict, "num_ctx": 4096},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]
