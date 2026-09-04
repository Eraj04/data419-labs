from pathlib import Path
import re

import kagglehub
import pandas as pd


KAGGLE_DATASET = "piterfm/fifa-football-world-cup"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _clean_columns(df):
    df = df.copy()

    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")
        for col in df.columns
    ]

    return df


def _download_csv(filename):
    path = kagglehub.dataset_download(
        KAGGLE_DATASET,
        path=filename,
    )

    return pd.read_csv(path)


def _prepare_matches():
    matches = _download_csv("matches_1930_2022.csv")
    matches = _clean_columns(matches)

    matches["date"] = pd.to_datetime(
        matches["date"],
        errors="coerce"
    )

    matches["total_goals"] = (
        matches["home_score"] + matches["away_score"]
    )

    matches["goal_difference"] = (
        matches["home_score"] - matches["away_score"]
    )

    return matches


def _prepare_tournaments():
    tournaments = _download_csv("world_cup.csv")
    tournaments = _clean_columns(tournaments)

    return tournaments


def _prepare_rankings():
    rankings = _download_csv("fifa_ranking_2022-10-06.csv")
    rankings = _clean_columns(rankings)

    return rankings


def pull_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    matches = _prepare_matches()
    tournaments = _prepare_tournaments()
    rankings = _prepare_rankings()

    matches.to_parquet(
        DATA_DIR / "matches.parquet",
        index=False
    )

    tournaments.to_parquet(
        DATA_DIR / "tournaments.parquet",
        index=False
    )

    rankings.to_parquet(
        DATA_DIR / "rankings.parquet",
        index=False
    )

    print(f"Saved {len(matches)} matches")
    print(f"Saved {len(tournaments)} tournaments")
    print(f"Saved {len(rankings)} rankings")


if __name__ == "__main__":
    pull_all()
