"""v2 deep-feature run: rich features + target framings, compared to v1."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import load_train
from .evaluate_v2 import leaderboard_v2

OUT = Path(__file__).resolve().parent.parent / "results"


def _compare(v1_path: Path, v2_best_rmse: float, label: str) -> None:
    if v1_path.exists():
        v1 = pd.read_csv(v1_path, index_col=0)
        best_v1 = v1["RMSE"].min()
        gain = (best_v1 - v2_best_rmse) / best_v1 * 100
        print(f"  {label}: best v1 RMSE={best_v1:.2f} -> best v2 RMSE={v2_best_rmse:.2f} "
              f"({gain:+.1f}% )")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train = load_train()

    print("== v2 FULL feature set (graded 12k model) ==")
    full = leaderboard_v2(train, feature_set="full")
    full.to_csv(OUT / "leaderboard_v2_full.csv", index=False)
    print("\n  Top 5 (full):")
    print(full.head(5).to_string(index=False))

    print("\n== v2 REDUCED feature set (December model) ==")
    reduced = leaderboard_v2(train, feature_set="reduced")
    reduced.to_csv(OUT / "leaderboard_v2_reduced.csv", index=False)
    print("\n  Top 5 (reduced):")
    print(reduced.head(5).to_string(index=False))

    print("\n== v1 -> v2 improvement ==")
    _compare(OUT / "leaderboard_v1_full.csv", full["RMSE"].min(), "FULL   ")
    _compare(OUT / "leaderboard_v1_reduced.csv", reduced["RMSE"].min(), "REDUCED")
    print(f"\nSaved v2 leaderboards to {OUT}")


if __name__ == "__main__":
    main()
