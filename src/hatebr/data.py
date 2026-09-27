"""Loading HateBR and producing ONE fixed train/test split shared by every approach."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.25
DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "HateBR.csv"


def load_hatebr(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load HateBR (Vargas et al., 2022). Uses the local CSV if present, else Hugging Face."""
    path = Path(csv_path) if csv_path else DEFAULT_CSV
    if path.exists():
        df = pd.read_csv(path)
    else:
        from datasets import load_dataset  # optional dependency

        df = load_dataset("franciellevargas/HateBR", split="train").to_pandas()
    df = df.rename(columns={"comentario": "text", "label_final": "label"})
    return df[["id", "text", "label", "links_post", "account_post"]].reset_index(drop=True)


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def fixed_split(df: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE) -> Split:
    """The single split used by rules, ML and LLM results (5,250 train / 1,750 test).

    Same parameters as the original Milestone 2/3 code, so the test set is the
    same 1,750 comments the original LLM experiments were scored on.
    """
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    return Split(train.reset_index(drop=True), test.reset_index(drop=True))
