"""
Join Bradley–Terry value scores with value metadata.

This script left-joins:
  * `bt_value_scores_full.csv` (or another scores CSV)
  * `labeled_topk_values.csv`   (value metadata)

on:
  scores.value_name == metadata.val

and writes a new CSV containing all score columns plus any matching
metadata columns. Values without metadata are kept with NaNs in the
metadata fields.
"""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd


def join_scores_with_metadata(
    scores_path: str,
    metadata_path: str,
    output_path: str,
) -> None:
    """Left-join BT scores with value metadata and write to CSV."""
    scores_df = pd.read_csv(scores_path)
    meta_df = pd.read_csv(metadata_path)

    # Expect columns: value_name in scores, val in metadata.
    if "value_name" not in scores_df.columns:
        raise ValueError("Scores CSV must contain a 'value_name' column.")
    if "val" not in meta_df.columns:
        raise ValueError("Metadata CSV must contain a 'val' column.")

    # Avoid duplicate column name conflicts except 'val' and 'freq'.
    # We keep all metadata columns; if there is a collision, suffix the metadata.
    overlap_cols: List[str] = [
        c for c in meta_df.columns if c in scores_df.columns and c not in {"val", "freq"}
    ]
    if overlap_cols:
        meta_df = meta_df.rename(
            columns={c: f"meta_{c}" for c in overlap_cols},
        )

    merged = scores_df.merge(
        meta_df,
        how="left",
        left_on="value_name",
        right_on="val",
    )

    merged.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Left-join BT value scores with labeled value metadata."
    )
    parser.add_argument(
        "--scores-csv",
        type=str,
        default="bt_value_scores_full.csv",
        help="Input CSV with BT scores (e.g. bt_value_scores_full.csv).",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="labeled_topk_values.csv",
        help="Input CSV with value metadata (e.g. labeled_topk_values.csv).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="bt_value_scores_with_meta.csv",
        help="Output CSV path for joined data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    join_scores_with_metadata(
        scores_path=args.scores_csv,
        metadata_path=args.metadata_csv,
        output_path=args.output_csv,
    )


if __name__ == "__main__":
    main()


