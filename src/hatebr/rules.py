"""Milestone 1: rule-based classifier.

The rules and weights are the ones from the original submission. The one change
that matters: the lexicon is now learned from the TRAINING split only and
scored on the held-out test split. The original code built the lexicon from all
7,000 comments and then scored those same 7,000 comments.
"""
from __future__ import annotations

import re
import string
from functools import lru_cache

import pandas as pd
from nltk import FreqDist, bigrams, trigrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_STOP = set(stopwords.words("portuguese"))
_PUNCT = set(string.punctuation)
_JUNK = {">", "<", "..", "...", "....", "``", "''", "@"}


def clean_tokens(text: str, lowercase: bool = False) -> list[str]:
    toks = word_tokenize(text.lower() if lowercase else text)
    return [
        t for t in toks
        if t not in _STOP and t not in _PUNCT and t not in _JUNK and len(t) > 1 and not t.isdigit()
    ]


def build_lexicon(train: pd.DataFrame, min_freq: int = 3, min_ratio: float = 3.0,
                  max_ratio: float = 1000.0) -> dict[str, set]:
    """Frequency-ratio lexicon. Must only ever see training data."""
    toks = train["text"].map(clean_tokens)
    off = [t for ts in toks[train["label"] == 1] for t in ts]
    non = [t for ts in toks[train["label"] == 0] for t in ts]
    f_off, f_non = FreqDist(off), FreqDist(non)
    n_off, n_non = sum(f_off.values()), sum(f_non.values())

    offensive, contextual, positive = set(), set(), set()
    for w in set(f_off) | set(f_non):
        a, b = f_off.get(w, 0), f_non.get(w, 0)
        if a + b < min_freq:
            continue
        ra, rb = a / n_off, b / n_non
        if rb == 0 and a > min_freq:
            offensive.add(w)
        elif ra == 0 and b > min_freq:
            positive.add(w)
        elif ra > 0 and rb > 0:
            if min_ratio < ra / rb < max_ratio:
                (offensive if ra / rb > 5 else contextual).add(w)
            elif min_ratio < rb / ra < max_ratio:
                positive.add(w)

    bg_off, bg_non = FreqDist(bigrams(off)), FreqDist(bigrams(non))
    off_bigrams = {bg for bg, f in bg_off.most_common(100) if f > 5 and f > 2 * bg_non.get(bg, 0)}
    off_trigrams = {tg for tg, f in FreqDist(trigrams(off)).most_common(50) if f > 3}
    return {
        "offensive": offensive, "contextual": contextual, "positive": positive,
        "bigrams": off_bigrams, "trigrams": off_trigrams,
    }


@lru_cache(maxsize=1)
def _spacy():
    import spacy
    return spacy.load("pt_core_news_sm")


class RuleClassifier:
    """13 weighted rules (lexicon, n-grams, intensifiers, negation, regex, emoji, NER, sentiment)."""

    INTENSIFIERS = {"muito", "demais", "total", "completo", "puro", "tão", "super", "mega",
                    "completamente", "totalmente", "extremamente"}
    NEGATIONS = {"não", "nem", "nunca", "jamais", "nada", "ninguém"}
    W = {"offensive": 3.0, "contextual": 1.5, "positive": 2.0, "bigram": 2.5, "trigram": 2.0,
         "intensifier": 1.5, "neg_positive": 2.0, "censored": 2.0, "angry_emoji": 1.5,
         "happy_emoji": 1.5}
    RX_CENSORED = re.compile(r"\b\w*[*#@]+\w*\b")
    RX_ANGRY = re.compile(r"[😡🤬😤💢🖕]")
    RX_HAPPY = re.compile(r"[😊😃😄😁💕❤️👍🙏💖😘]")

    def __init__(self, lexicon: dict[str, set], use_ner: bool = True, use_sentiment: bool = True):
        self.lx = lexicon
        self.use_ner = use_ner
        self.use_sentiment = use_sentiment

    def score(self, text: str) -> tuple[float, float, list[str]]:
        toks = [t for t in word_tokenize(text.lower()) if t not in _PUNCT and len(t) > 1]
        off = [t for t in toks if t in self.lx["offensive"]]
        ctx = [t for t in toks if t in self.lx["contextual"]]
        pos = [t for t in toks if t in self.lx["positive"]]
        bgs = [b for b in bigrams(toks) if b in self.lx["bigrams"]]
        tgs = [g for g in trigrams(toks) if g in self.lx["trigrams"]]
        s_off = s_non = 0.0
        fired: list[str] = []

        def hit(rule: str, delta_off: float = 0.0, delta_non: float = 0.0):
            nonlocal s_off, s_non
            s_off += delta_off
            s_non += delta_non
            fired.append(rule)

        if off: hit("R1_offensive_word", len(off) * self.W["offensive"])
        if ctx: hit("R2_contextual_word", len(ctx) * self.W["contextual"])
        if bgs: hit("R3_offensive_bigram", len(bgs) * self.W["bigram"])
        if tgs: hit("R4_offensive_trigram", len(tgs) * self.W["trigram"])
        if off and any(t in self.INTENSIFIERS for t in toks):
            hit("R5_intensifier_plus_offense", self.W["intensifier"])
        if pos and any(t in self.NEGATIONS for t in toks):
            hit("R6_negation_plus_positive", self.W["neg_positive"])
        if self.RX_CENSORED.search(text): hit("R7_censored_swear", self.W["censored"])
        n_angry = len(self.RX_ANGRY.findall(text))
        if n_angry: hit("R8_angry_emoji", n_angry * self.W["angry_emoji"])
        if pos: hit("R9_positive_word", delta_non=len(pos) * self.W["positive"])
        n_happy = len(self.RX_HAPPY.findall(text))
        if n_happy: hit("R10_positive_emoji", delta_non=n_happy * self.W["happy_emoji"])
        if len(toks) > 20 and not off: hit("R11_long_without_offense", delta_non=1.0)
        if self.use_ner and off and any(e.label_ == "PER" for e in _spacy()(text).ents):
            hit("R12_person_plus_offense", 2.0)
        if self.use_sentiment:
            from textblob import TextBlob
            if TextBlob(text).sentiment.polarity < -0.3:
                hit("R13_negative_sentiment", 1.5)
        return s_off, s_non, fired

    def predict_one(self, text: str) -> int:
        s_off, s_non, fired = self.score(text)
        return int(bool(fired) and s_off > s_non + 0.5)

    def predict(self, texts) -> list[int]:
        return [self.predict_one(t) for t in texts]
