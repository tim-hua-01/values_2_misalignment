"""
Bradley–Terry value ranking with nudge adjustment, operating directly on parquet files.

This script:
  * Reads only the needed columns from `raw_data/data/*.parquet` via DuckDB.
  * For each model, builds pairwise comparisons and fits a generalized Bradley–Terry model:
        eta_i = alpha + (theta[v1] - theta[v2]) + beta * bias_i
  * Outputs per-value scores and frequencies for each model.

Usage (small-sample test first):

    uv run python bt_rank_values.py \
        --parquet-glob "raw_data/data/*.parquet" \
        --sample-frac 0.001 \
        --max-rows 100000 \
        --output-csv "bt_value_scores_sample.csv"

Then scale up to full data by increasing --sample-frac (e.g. 0.01, then 1.0) and/or removing --max-rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd


MODEL_NAMES: List[str] = [
    "claude_3_5_sonnet",
    "claude_3_7_sonnet",
    "claude_opus_3",
    "claude_opus_4",
    "claude_sonnet_4",
    "gemini_2_5_pro",
    "gpt_4_1",
    "gpt_4_1_mini",
    "gpt_4o",
    "grok_4",
    "o3",
    "o4_mini",
]


@dataclass
class BTResult:
    model: str
    theta: np.ndarray  # full-length over global value index, NaN for inactive values
    freq: np.ndarray   # per-value comparison counts for this model
    alpha: float
    beta: float


def _escape_single_quotes(path: str) -> str:
    """Escape single quotes for safe inclusion in a DuckDB SQL string literal."""
    return path.replace("'", "''")


def build_value_mapping(con: duckdb.DuckDBPyConnection, parquet_glob: str) -> Tuple[Dict[str, int], List[str]]:
    """
    Build a global mapping from value string to integer index based on all value1/value2 in the data.
    """
    glob_escaped = _escape_single_quotes(parquet_glob)
    query = f"""
        WITH vals AS (
            SELECT value1 AS val FROM parquet_scan('{glob_escaped}')
            UNION
            SELECT value2 AS val FROM parquet_scan('{glob_escaped}')
        )
        SELECT DISTINCT val
        FROM vals
        WHERE val IS NOT NULL
        ORDER BY val
    """
    df = con.execute(query).df()
    if df.empty:
        raise RuntimeError("No values found in value1/value2 columns when building mapping.")

    values: List[str] = df["val"].tolist()
    value_to_idx: Dict[str, int] = {v: i for i, v in enumerate(values)}
    return value_to_idx, values


def prepare_model_data(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    model: str,
    value_to_idx: Dict[str, int],
    sample_frac: float,
    max_rows: int | None,
    min_value_comparisons: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a single model, extract comparison rows and construct arrays:
        v1_idx, v2_idx, y, bias, active_value_indices
    where:
        y    = 1 if model favored value1, 0 if favored value2 (ties removed)
        bias = +1 for nudge_direction == "value1", -1 for "value2", 0 otherwise

    active_value_indices maps local parameter indices back to global value indices.
    """
    v1_col = f"{model}_value1_position"
    v2_col = f"{model}_value2_position"

    glob_escaped = _escape_single_quotes(parquet_glob)

    where_clauses = [
        f"{v1_col} IS NOT NULL",
        f"{v2_col} IS NOT NULL",
    ]
    if 0.0 < sample_frac < 1.0:
        where_clauses.append(f"random() < {sample_frac}")
    where_sql = " AND ".join(where_clauses)

    query = f"""
        SELECT
            value1,
            value2,
            nudge_direction,
            {v1_col} AS s1,
            {v2_col} AS s2
        FROM parquet_scan('{glob_escaped}')
        WHERE {where_sql}
    """
    if max_rows is not None and max_rows > 0:
        query += f"\nLIMIT {max_rows}"

    df = con.execute(query).df()
    if df.empty:
        raise RuntimeError(f"No rows found for model '{model}' after initial filtering.")

    # Remove rows with missing values (paranoia) and ties.
    df = df.dropna(subset=["value1", "value2", "s1", "s2"])
    if df.empty:
        raise RuntimeError(f"All rows for model '{model}' were dropped due to missing values.")

    s1 = df["s1"].to_numpy()
    s2 = df["s2"].to_numpy()
    neq_mask = s1 != s2
    df = df.loc[neq_mask].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"All rows for model '{model}' are ties (s1 == s2) after filtering.")

    s1 = df["s1"].to_numpy()
    s2 = df["s2"].to_numpy()
    y = (s1 > s2).astype(np.float64)

    # Compute bias from nudge_direction.
    nd = df["nudge_direction"].astype("string")
    bias = np.zeros(len(df), dtype=np.float64)
    bias[nd == "value1"] = 1.0
    bias[nd == "value2"] = -1.0

    # Map values to global indices.
    v1_idx = df["value1"].map(value_to_idx)
    v2_idx = df["value2"].map(value_to_idx)
    if v1_idx.isna().any() or v2_idx.isna().any():
        raise RuntimeError(f"Encountered value strings for model '{model}' that are missing in value_to_idx mapping.")

    v1_idx_arr = v1_idx.to_numpy(dtype=np.int32)
    v2_idx_arr = v2_idx.to_numpy(dtype=np.int32)

    if min_value_comparisons > 0:
        # Filter out very rare values for this model to stabilize estimates.
        counts = np.bincount(
            np.concatenate([v1_idx_arr, v2_idx_arr]),
            minlength=len(value_to_idx),
        )
        keep_mask_values = counts >= min_value_comparisons
        row_keep_mask = keep_mask_values[v1_idx_arr] & keep_mask_values[v2_idx_arr]
        v1_idx_arr = v1_idx_arr[row_keep_mask]
        v2_idx_arr = v2_idx_arr[row_keep_mask]
        y = y[row_keep_mask]
        bias = bias[row_keep_mask]

        if v1_idx_arr.size == 0:
            raise RuntimeError(
                f"After applying min_value_comparisons={min_value_comparisons}, "
                f"no rows remain for model '{model}'."
            )

    # Build list of active global value indices for this model.
    active_values = np.unique(np.concatenate([v1_idx_arr, v2_idx_arr]))
    return v1_idx_arr, v2_idx_arr, y, bias, active_values


def fit_bt_model(
    num_values: int,
    v1_idx: np.ndarray,
    v2_idx: np.ndarray,
    y: np.ndarray,
    bias: np.ndarray,
    l2_reg: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit generalized Bradley–Terry parameters (theta, alpha, beta) via gradient ascent.

    num_values: number of distinct values in this model's active set.
    v1_idx, v2_idx: arrays of indices in [0, num_values).
    y: binary outcomes (1 if value1 preferred, 0 if value2 preferred).
    bias: nudge bias scalar per comparison (-1, 0, +1).
    """
    theta = np.zeros(num_values, dtype=np.float64)
    alpha = 0.0
    beta = 0.0

    prev_ll = -np.inf

    for iteration in range(1, max_iter + 1):
        # Linear predictor and probability.
        eta = alpha + (theta[v1_idx] - theta[v2_idx]) + beta * bias
        # Clip to avoid numerical overflow in exp.
        eta = np.clip(eta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))

        # Gradient of log-likelihood w.r.t. eta is (y - p).
        err = y - p

        # Gradients for theta using incidence of values in v1 and v2.
        grad_theta = np.zeros_like(theta)
        np.add.at(grad_theta, v1_idx, err)
        np.add.at(grad_theta, v2_idx, -err)

        # Gradients for alpha and beta.
        grad_alpha = float(err.sum())
        grad_beta = float(np.dot(err, bias))

        # L2 regularization on theta.
        grad_theta -= l2_reg * theta

        # Gradient ascent update.
        theta += learning_rate * grad_theta
        alpha += learning_rate * grad_alpha
        beta += learning_rate * grad_beta

        # Enforce identifiability: center theta so that its mean is zero.
        theta -= float(theta.mean())

        # Regularized log-likelihood for convergence monitoring.
        ll = (
            (y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12)).sum()
            - 0.5 * l2_reg * float((theta ** 2).sum())
        )

        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return theta, alpha, beta


def run_pipeline(
    parquet_glob: str,
    output_csv: str,
    sample_frac: float,
    max_rows: int | None,
    min_value_comparisons: int,
    l2_reg: float,
    learning_rate: float,
    max_iter: int,
    tol: float,
) -> None:
    """
    Orchestrate data loading, per-model fitting, and result aggregation.
    """
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}.")

    con = duckdb.connect()  # in-memory is fine; we only query parquet_scan

    print("Building global value mapping from parquet data...")
    value_to_idx, idx_to_value = build_value_mapping(con, parquet_glob)
    num_values_global = len(value_to_idx)
    print(f"Found {num_values_global} distinct values.")

    bt_results: List[BTResult] = []

    for model in MODEL_NAMES:
        print(f"\n=== Fitting model: {model} ===")
        v1_idx, v2_idx, y, bias, active_values = prepare_model_data(
            con=con,
            parquet_glob=parquet_glob,
            model=model,
            value_to_idx=value_to_idx,
            sample_frac=sample_frac,
            max_rows=max_rows,
            min_value_comparisons=min_value_comparisons,
        )

        # Remap global indices in active_values to a dense 0..K-1 range for this model.
        active_values = np.asarray(active_values, dtype=np.int32)
        local_index_for_global = {int(g): int(i) for i, g in enumerate(active_values)}

        v1_local = np.fromiter(
            (local_index_for_global[int(g)] for g in v1_idx),
            dtype=np.int32,
            count=len(v1_idx),
        )
        v2_local = np.fromiter(
            (local_index_for_global[int(g)] for g in v2_idx),
            dtype=np.int32,
            count=len(v2_idx),
        )

        print(
            f"Number of comparisons for {model}: {len(y)} "
            f"(active values: {len(active_values)})"
        )

        theta_local, alpha, beta = fit_bt_model(
            num_values=len(active_values),
            v1_idx=v1_local,
            v2_idx=v2_local,
            y=y,
            bias=bias,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
        )

        # Expand local theta into global-length array with NaN for inactive values.
        theta_global = np.full(num_values_global, np.nan, dtype=np.float64)
        theta_global[active_values] = theta_local

        # Per-value frequency for this model.
        freq_global = np.bincount(
            np.concatenate([v1_idx, v2_idx]),
            minlength=num_values_global,
        )

        bt_results.append(
            BTResult(
                model=model,
                theta=theta_global,
                freq=freq_global,
                alpha=alpha,
                beta=beta,
            )
        )

    # Aggregate results into a long table: one row per (model, value).
    records: List[Dict[str, object]] = []
    for res in bt_results:
        model = res.model
        theta = res.theta
        freq = res.freq

        for idx, (val_name, theta_v, freq_v) in enumerate(
            zip(idx_to_value, theta, freq, strict=True)
        ):
            if np.isnan(theta_v) or freq_v == 0:
                continue
            records.append(
                {
                    "model": model,
                    "value_index": idx,
                    "value_name": val_name,
                    "theta": float(theta_v),
                    "freq": int(freq_v),
                }
            )

    if not records:
        raise RuntimeError("No records generated; check filtering settings and input data.")

    result_df = pd.DataFrame.from_records(records)

    # Add per-model ranks (1 = highest theta).
    result_df["rank"] = (
        result_df.groupby("model")["theta"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    # Sort for readability: by model, then rank.
    result_df = result_df.sort_values(["model", "rank", "value_name"]).reset_index(drop=True)

    print(f"\nWriting results to {output_csv} ...")
    result_df.to_csv(output_csv, index=False)
    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Bradley–Terry value scores with nudge adjustment for each model."
    )
    parser.add_argument(
        "--parquet-glob",
        type=str,
        default="raw_data/data/*.parquet",
        help="Glob pattern to parquet files containing comparisons data.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="bt_value_scores.csv",
        help="Output CSV path for per-model value scores.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.001,
        help="Fraction of rows to sample per model (0 < f <= 1). "
        "Start small (e.g. 0.001), then increase to 1.0 for full data.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100_000,
        help="Maximum number of rows per model after sampling (useful for quick tests). "
        "Set to 0 or negative to disable the limit.",
    )
    parser.add_argument(
        "--min-value-comparisons",
        type=int,
        default=5,
        help="Minimum number of comparisons a value must appear in (per model) "
        "to be included in fitting.",
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=1e-2,
        help="L2 regularization strength for theta.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for gradient ascent.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum number of gradient ascent iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Convergence tolerance on change in regularized log-likelihood.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None

    run_pipeline(
        parquet_glob=args.parquet_glob,
        output_csv=args.output_csv,
        sample_frac=args.sample_frac,
        max_rows=max_rows,
        min_value_comparisons=args.min_value_comparisons,
        l2_reg=args.l2_reg,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        tol=args.tol,
    )


if __name__ == "__main__":
    main()


