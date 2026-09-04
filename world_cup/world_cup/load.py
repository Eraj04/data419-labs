from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load():
    matches = pd.read_parquet(DATA_DIR / "matches.parquet")
    tournaments = pd.read_parquet(DATA_DIR / "tournaments.parquet")
    rankings = pd.read_parquet(DATA_DIR / "rankings.parquet")

    return {
        "matches": matches,
        "tournaments": tournaments,
        "rankings": rankings,
    }

