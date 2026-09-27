"""Guards for the three mistakes this repo corrects."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hatebr.data import fixed_split, load_hatebr  # noqa: E402
from hatebr.llm import STRATEGIES, parse, pick_few_shot, validation_sample  # noqa: E402
from hatebr.rules import build_lexicon  # noqa: E402


@pytest.fixture(scope="module")
def split():
    return fixed_split(load_hatebr())


def test_split_is_fixed_and_disjoint(split):
    assert len(split.train) == 5250 and len(split.test) == 1750
    assert set(split.train["id"]).isdisjoint(split.test["id"])
    assert split.test["label"].mean() == pytest.approx(0.5)


def test_lexicon_ignores_test_data(split):
    """Adding the test labels must not be able to change a lexicon built from train only."""
    lex = build_lexicon(split.train)
    shuffled = split.test.assign(label=1 - split.test["label"])
    assert build_lexicon(split.train) == lex
    assert build_lexicon(pd.concat([split.train, shuffled])) != lex  # sanity: data does matter


def test_few_shot_prompt_contains_the_examples(split):
    ex = pick_few_shot(split.train)
    assert len(ex) == 16 and sum(e["label"] for e in ex) == 8
    assert set(ex_id for ex_id in [e["text"] for e in ex]) <= set(split.train["text"])
    for name in ("fs-original", "fs-rewrite"):
        prompt = STRATEGIES[name].build("texto qualquer", ex)
        assert all(e["text"] in prompt for e in ex)
        assert prompt.rstrip().endswith("Classificação:")


def test_prompt_selection_never_touches_test(split):
    val = validation_sample(split.train)
    assert len(val) == 300 and val["label"].mean() == 0.5
    assert set(val["id"]).isdisjoint(split.test["id"])
    assert set(val["id"]).isdisjoint({e["id"] for e in pick_few_shot(split.train)})


@pytest.mark.parametrize("answer,strategy,expected", [
    ("1", "label", 1),
    (" 0\n0\n", "label", 0),
    ("Desculpe, não posso classificar este conteúdo.", "label", None),
    ("Lamento, mas não posso ajudar com isso.", "cot", None),
    ("Há insulto (1). Mas é crítica.\nRESPOSTA: 0", "cot", 0),
    ("1. Não há palavrões.\n2. Não há ataque.\n3. É crítica dura", "cot", None),  # truncated: no label
])
def test_parse_keeps_refusals_as_abstentions(answer, strategy, expected):
    assert parse(answer, strategy) == expected
