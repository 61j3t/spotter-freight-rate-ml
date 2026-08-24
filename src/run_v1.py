"""v1 baseline: run the full roster on time-based CV and print a leaderboard.

Two boards:
  - FULL feature set   -> proxy for the graded 12k validation model.
  - REDUCED feature set -> proxy for the December (shared-columns) model.

This is the "before feature engineering" baseline we compare v2 against.
"""
from __future__ import annotations

from pathlib import Path

from .data import load_train
from .evaluate import leaderboard

OUT = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train = load_train()
    print(f"Loaded {len(train):,} training rows ({train['date'].min().date()} "
          f"-> {train['date'].max().date()})\n")

    print("== FULL feature set (graded 12k model) ==")
    full = leaderboard(train, feature_set="full")
    full.to_csv(OUT / "leaderboard_v1_full.csv")

    print("\n== REDUCED feature set (December model) ==")
    reduced = leaderboard(train, feature_set="reduced")
    reduced.to_csv(OUT / "leaderboard_v1_reduced.csv")

    print(f"\nSaved leaderboards to {OUT}")


if __name__ == "__main__":
    main()
