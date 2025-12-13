"""
Bradley–Terry value ranking with nudge adjustment, operating directly on parquet files.

This script:
  * Reads only the needed columns from `raw_data/data/*.parquet` via DuckDB.
  * For each model, builds pairwise comparisons and fits a generalized Bradley–Terry model:
        eta_i = alpha + (theta[v1] - theta[v2]) + beta * bias_i
  * Uses scipy's L-BFGS-B optimizer for fast, robust convergence.
  * Outputs per-value scores and frequencies for each model.
  * Automatically generates both unmerged and metadata-joined versions when --metadata-csv is provided.

Usage (small-sample test first):

    uv run python bt_rank_values.py \
        --parquet-glob "raw_data/data/*.parquet" \
        --sample-frac 0.001 \
        --max-rows 100000 \
        --seed 42 \
        --output-csv "bt_value_scores_sample.csv"

Then scale up to full data by increasing --sample-frac (e.g. 0.01, then 1.0) and/or removing --max-rows.

With metadata joining (outputs both unmerged and merged versions):

    uv run python bt_rank_values.py \
        --parquet-glob "raw_data/data/*.parquet" \
        --sample-frac 1.0 \
        --output-csv "bt_value_scores_full.csv" \
        --metadata-csv "labeled_topk_values.csv"

This creates both bt_value_scores_full.csv and bt_value_scores_full_with_meta.csv.

Validation mode (bootstrap held-out accuracy, separate models per AI):

    uv run python bt_rank_values.py \
        --parquet-glob "raw_data/data/*.parquet" \
        --validate \
        --n-bootstrap 50 \
        --test-frac 0.1 \
        --seed 42

Pooled validation mode (one universal BT model, per-AI test accuracy):

    uv run python bt_rank_values.py \
        --parquet-glob "raw_data/data/*.parquet" \
        --validate-pooled \
        --n-bootstrap 50 \
        --test-frac 0.1 \
        --seed 42

Aggregated values mode (use merged value names):

uv run python bt_rank_values.py \
    --parquet-glob "raw_data/data/*.parquet" \
    --aggregate \
    --sample-frac 1.0 \
    --max-rows 0 \
    --output-csv "bt_value_scores_aggregated.csv"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize


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


def load_aggregation_mapping(csv_path: str) -> Dict[str, str]:
    """
    Load value aggregation mapping from CSV.
    Returns dict mapping original value_name -> merged_value_names.
    """
    df = pd.read_csv(csv_path)
    if 'value_name' not in df.columns or 'merged_value_names' not in df.columns:
        raise ValueError(f"CSV {csv_path} must have 'value_name' and 'merged_value_names' columns")
    
    mapping = dict(zip(df['value_name'], df['merged_value_names']))
    print(f"Loaded aggregation mapping with {len(mapping)} values from {csv_path}")
    return mapping


def build_value_mapping(
    con: duckdb.DuckDBPyConnection, 
    parquet_glob: str,
    aggregation_map: Dict[str, str] | None = None,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Build a global mapping from value string to integer index based on all value1/value2 in the data.
    
    If aggregation_map is provided, values are mapped to their merged versions first,
    then the mapping is built over the merged values.
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
    
    # Apply aggregation if requested
    if aggregation_map is not None:
        values = [aggregation_map.get(v, v) for v in values]
        # Get unique merged values
        values = sorted(set(values))
        print(f"After aggregation: {len(values)} unique merged values")
    
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
    seed: int | None,
    aggregation_map: Dict[str, str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a single model, extract comparison rows and construct arrays:
        v1_idx, v2_idx, y, bias, active_value_indices
    where:
        y    = 1 if model favored value1, 0 if favored value2 (ties removed)
        bias = +1 for nudge_direction == "value1", -1 for "value2", 0 otherwise

    active_value_indices maps local parameter indices back to global value indices.
    
    If aggregation_map is provided, values are mapped to their merged versions and
    comparisons where merged_value1 == merged_value2 are dropped.
    """
    v1_col = f"{model}_value1_position"
    v2_col = f"{model}_value2_position"

    glob_escaped = _escape_single_quotes(parquet_glob)

    where_clauses = [
        f"{v1_col} IS NOT NULL",
        f"{v2_col} IS NOT NULL",
    ]
    if 0.0 < sample_frac < 1.0:
        # Set DuckDB seed for reproducible sampling if provided
        if seed is not None:
            con.execute(f"SELECT setseed({seed / 2**31})")  # setseed takes value in [0, 1]
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

    # Apply aggregation if requested
    if aggregation_map is not None:
        df["value1"] = df["value1"].map(lambda v: aggregation_map.get(v, v))
        df["value2"] = df["value2"].map(lambda v: aggregation_map.get(v, v))
        
        # Remove rows where aggregated values are the same
        same_value_mask = df["value1"] == df["value2"]
        n_dropped = same_value_mask.sum()
        if n_dropped > 0:
            df = df.loc[~same_value_mask].reset_index(drop=True)
            print(f"  Dropped {n_dropped} rows where aggregated values were identical")
        
        if df.empty:
            raise RuntimeError(f"All rows for model '{model}' were dropped after aggregation (all same-value comparisons).")

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
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, float, float, dict]:
    """
    Fit generalized Bradley–Terry parameters (theta, alpha, beta) via L-BFGS-B.

    num_values: number of distinct values in this model's active set.
    v1_idx, v2_idx: arrays of indices in [0, num_values).
    y: binary outcomes (1 if value1 preferred, 0 if value2 preferred).
    bias: nudge bias scalar per comparison (-1, 0, +1).
    
    Returns (theta, alpha, beta, opt_info) where opt_info contains optimization details.
    """
    # Parameter layout: [theta_0, ..., theta_{K-2}, alpha, beta]
    # We fix theta_{K-1} = 0 for identifiability (sum-to-zero equivalent via one constraint)
    n_theta_free = num_values - 1  # last theta fixed at 0
    n_params = n_theta_free + 2  # theta_free + alpha + beta
    
    def neg_log_likelihood_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute negative regularized log-likelihood and gradient."""
        theta_free = params[:n_theta_free]
        alpha = params[n_theta_free]
        beta = params[n_theta_free + 1]
        
        # Full theta with last element fixed to 0
        theta = np.zeros(num_values, dtype=np.float64)
        theta[:n_theta_free] = theta_free
        
        # Linear predictor and probability
        eta = alpha + (theta[v1_idx] - theta[v2_idx]) + beta * bias
        eta = np.clip(eta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        
        # Negative log-likelihood (we minimize)
        nll = -(y * np.log(p + 1e-12) + (1.0 - y) * np.log(1.0 - p + 1e-12)).sum()
        # Add L2 regularization on theta
        nll += 0.5 * l2_reg * float((theta_free ** 2).sum())
        
        # Gradient of NLL w.r.t. eta is (p - y)
        err = p - y  # note: flipped sign since we minimize
        
        # Gradient for full theta
        grad_theta_full = np.zeros(num_values, dtype=np.float64)
        np.add.at(grad_theta_full, v1_idx, err)
        np.add.at(grad_theta_full, v2_idx, -err)
        
        # Extract gradient for free theta parameters only
        grad_theta_free = grad_theta_full[:n_theta_free]
        grad_theta_free += l2_reg * theta_free  # L2 regularization gradient
        
        grad_alpha = float(err.sum())
        grad_beta = float(np.dot(err, bias))
        
        grad = np.concatenate([grad_theta_free, [grad_alpha, grad_beta]])
        return nll, grad
    
    # Initial parameters
    x0 = np.zeros(n_params, dtype=np.float64)
    
    # Optimize using L-BFGS-B
    result = minimize(
        neg_log_likelihood_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,  # function returns (value, gradient)
        options={"maxiter": max_iter, "ftol": tol, "gtol": 1e-6},
    )
    
    # Extract parameters
    theta_free = result.x[:n_theta_free]
    alpha = result.x[n_theta_free]
    beta = result.x[n_theta_free + 1]
    
    # Reconstruct full theta and center it (mean = 0)
    theta = np.zeros(num_values, dtype=np.float64)
    theta[:n_theta_free] = theta_free
    theta -= theta.mean()
    
    opt_info = {
        "success": result.success,
        "message": result.message,
        "n_iter": result.nit,
        "final_nll": result.fun,
    }
    
    return theta, alpha, beta, opt_info


def load_model_data_raw(
    con: duckdb.DuckDBPyConnection,
    parquet_glob: str,
    model: str,
    value_to_idx: Dict[str, int],
    aggregation_map: Dict[str, str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ALL comparison data for a model without any filtering (except ties).
    Returns (v1_idx, v2_idx, y, bias) as global indices.
    
    If aggregation_map is provided, values are mapped to their merged versions and
    comparisons where merged_value1 == merged_value2 are dropped.
    """
    v1_col = f"{model}_value1_position"
    v2_col = f"{model}_value2_position"
    glob_escaped = _escape_single_quotes(parquet_glob)

    query = f"""
        SELECT
            value1,
            value2,
            nudge_direction,
            {v1_col} AS s1,
            {v2_col} AS s2
        FROM parquet_scan('{glob_escaped}')
        WHERE {v1_col} IS NOT NULL AND {v2_col} IS NOT NULL
    """
    df = con.execute(query).df()
    if df.empty:
        raise RuntimeError(f"No rows found for model '{model}'.")

    # Apply aggregation if requested
    if aggregation_map is not None:
        df["value1"] = df["value1"].map(lambda v: aggregation_map.get(v, v))
        df["value2"] = df["value2"].map(lambda v: aggregation_map.get(v, v))
        
        # Remove rows where aggregated values are the same
        same_value_mask = df["value1"] == df["value2"]
        if same_value_mask.any():
            df = df.loc[~same_value_mask].reset_index(drop=True)
        
        if df.empty:
            raise RuntimeError(f"All rows for model '{model}' were dropped after aggregation.")

    # Remove missing values and ties
    df = df.dropna(subset=["value1", "value2", "s1", "s2"])
    s1 = df["s1"].to_numpy()
    s2 = df["s2"].to_numpy()
    neq_mask = s1 != s2
    df = df.loc[neq_mask].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"All rows for model '{model}' are ties after filtering.")

    s1 = df["s1"].to_numpy()
    s2 = df["s2"].to_numpy()
    y = (s1 > s2).astype(np.float64)

    # Compute bias from nudge_direction
    nd = df["nudge_direction"].astype("string")
    bias = np.zeros(len(df), dtype=np.float64)
    bias[nd == "value1"] = 1.0
    bias[nd == "value2"] = -1.0

    # Map values to global indices
    v1_idx = df["value1"].map(value_to_idx).to_numpy(dtype=np.int32)
    v2_idx = df["value2"].map(value_to_idx).to_numpy(dtype=np.int32)

    return v1_idx, v2_idx, y, bias


def run_validation_pooled(
    parquet_glob: str,
    n_bootstrap: int,
    test_frac: float,
    min_value_comparisons: int,
    min_high_conf_comparisons: int,
    l2_reg: float,
    max_iter: int,
    tol: float,
    seed: int | None,
    validation_output_csv: str | None,
    aggregation_map: Dict[str, str] | None = None,
) -> None:
    """
    Run bootstrap validation with POOLED model: fit one BT model on all models' training data,
    then evaluate on each model's test set separately.
    
    This tests whether a universal value preference model generalizes to individual AI systems.
    """
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed set to {seed}")

    con = duckdb.connect()

    print("Building global value mapping from parquet data...")
    value_to_idx, idx_to_value = build_value_mapping(con, parquet_glob, aggregation_map)
    num_values_global = len(value_to_idx)
    print(f"Found {num_values_global} distinct values.")

    # Store results per model: {model: {"accuracy": [...], "log_loss": [...], ...}}
    all_results: Dict[str, Dict[str, List[float]]] = {
        model: {
            "accuracy": [], "log_loss": [], "n_test": [], "n_train": [],
            "accuracy_high_conf": [], "log_loss_high_conf": [], "n_test_high_conf": [],
        }
        for model in MODEL_NAMES
    }

    print(f"\nRunning {n_bootstrap} bootstrap iterations with POOLED model (test_frac={test_frac})...")
    print(f"High-confidence threshold: {min_high_conf_comparisons} train comparisons per value")
    print("=" * 70)

    for bootstrap_iter in range(n_bootstrap):
        print(f"\n--- Bootstrap iteration {bootstrap_iter + 1}/{n_bootstrap} ---")

        # Load all models' data and split into train/test
        model_splits: Dict[str, Dict[str, np.ndarray]] = {}
        
        for model in MODEL_NAMES:
            # Load all data for this model
            v1_idx, v2_idx, y, bias = load_model_data_raw(con, parquet_glob, model, value_to_idx, aggregation_map)
            n_total = len(y)

            # Random train/test split
            perm = np.random.permutation(n_total)
            n_test = int(n_total * test_frac)
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]

            model_splits[model] = {
                "v1_train": v1_idx[train_idx],
                "v2_train": v2_idx[train_idx],
                "y_train": y[train_idx],
                "bias_train": bias[train_idx],
                "v1_test": v1_idx[test_idx],
                "v2_test": v2_idx[test_idx],
                "y_test": y[test_idx],
                "bias_test": bias[test_idx],
            }

        # Pool all training data across models
        v1_train_pooled = np.concatenate([model_splits[m]["v1_train"] for m in MODEL_NAMES])
        v2_train_pooled = np.concatenate([model_splits[m]["v2_train"] for m in MODEL_NAMES])
        y_train_pooled = np.concatenate([model_splits[m]["y_train"] for m in MODEL_NAMES])
        bias_train_pooled = np.concatenate([model_splits[m]["bias_train"] for m in MODEL_NAMES])

        # Apply min_value_comparisons filter on POOLED TRAIN set
        counts_train = np.bincount(
            np.concatenate([v1_train_pooled, v2_train_pooled]),
            minlength=num_values_global,
        )
        fitted_values_mask = counts_train >= min_value_comparisons
        fitted_values = np.where(fitted_values_mask)[0]

        # Filter pooled train data
        train_row_mask = fitted_values_mask[v1_train_pooled] & fitted_values_mask[v2_train_pooled]
        v1_train_pooled = v1_train_pooled[train_row_mask]
        v2_train_pooled = v2_train_pooled[train_row_mask]
        y_train_pooled = y_train_pooled[train_row_mask]
        bias_train_pooled = bias_train_pooled[train_row_mask]

        if len(v1_train_pooled) == 0:
            print("  No pooled train data after filtering, skipping iteration.")
            continue

        # Remap to local indices for fitting
        local_index_for_global = {int(g): int(i) for i, g in enumerate(fitted_values)}
        v1_train_local = np.array([local_index_for_global[g] for g in v1_train_pooled], dtype=np.int32)
        v2_train_local = np.array([local_index_for_global[g] for g in v2_train_pooled], dtype=np.int32)

        # Fit ONE model on pooled training data
        print(f"  Fitting pooled model on {len(y_train_pooled)} comparisons ({len(fitted_values)} values)...")
        theta_local, alpha, beta, opt_info = fit_bt_model(
            num_values=len(fitted_values),
            v1_idx=v1_train_local,
            v2_idx=v2_train_local,
            y=y_train_pooled,
            bias=bias_train_pooled,
            l2_reg=l2_reg,
            max_iter=max_iter,
            tol=tol,
        )

        status = "converged" if opt_info["success"] else "did NOT converge"
        print(f"  Pooled model {status} in {opt_info['n_iter']} iterations (nll={opt_info['final_nll']:.4f})")

        # Expand theta to global indices
        theta_global = np.full(num_values_global, np.nan, dtype=np.float64)
        theta_global[fitted_values] = theta_local

        # Now evaluate on each model's test set
        for model in MODEL_NAMES:
            v1_test = model_splits[model]["v1_test"]
            v2_test = model_splits[model]["v2_test"]
            y_test = model_splits[model]["y_test"]
            bias_test = model_splits[model]["bias_test"]

            # Filter test to only include comparisons where BOTH values were fitted
            test_row_mask = fitted_values_mask[v1_test] & fitted_values_mask[v2_test]
            v1_test_filt = v1_test[test_row_mask]
            v2_test_filt = v2_test[test_row_mask]
            y_test_filt = y_test[test_row_mask]
            bias_test_filt = bias_test[test_row_mask]

            if len(y_test_filt) == 0:
                print(f"  {model}: No test data for fitted values, skipping.")
                continue

            # Count train comparisons per value for this model (for high-conf calculation)
            v1_train_model = model_splits[model]["v1_train"]
            v2_train_model = model_splits[model]["v2_train"]
            counts_train_model = np.bincount(
                np.concatenate([v1_train_model, v2_train_model]),
                minlength=num_values_global,
            )

            # Predict on test set using pooled model
            eta_test = alpha + (theta_global[v1_test_filt] - theta_global[v2_test_filt]) + beta * bias_test_filt
            eta_test = np.clip(eta_test, -30.0, 30.0)
            p_test = 1.0 / (1.0 + np.exp(-eta_test))

            # Compute metrics (all test data)
            pred = (p_test > 0.5).astype(float)
            accuracy = float((pred == y_test_filt).mean())
            log_loss = float(-(y_test_filt * np.log(p_test + 1e-12) + (1 - y_test_filt) * np.log(1 - p_test + 1e-12)).mean())

            all_results[model]["accuracy"].append(accuracy)
            all_results[model]["log_loss"].append(log_loss)
            all_results[model]["n_test"].append(len(y_test_filt))
            all_results[model]["n_train"].append(len(model_splits[model]["y_train"]))

            # High-confidence metrics: test pairs where BOTH values had >= min_high_conf_comparisons in THIS MODEL's train set
            high_conf_mask = (counts_train_model[v1_test_filt] >= min_high_conf_comparisons) & (counts_train_model[v2_test_filt] >= min_high_conf_comparisons)
            n_high_conf = high_conf_mask.sum()

            if n_high_conf > 0:
                pred_hc = pred[high_conf_mask]
                y_test_hc = y_test_filt[high_conf_mask]
                p_test_hc = p_test[high_conf_mask]

                accuracy_hc = float((pred_hc == y_test_hc).mean())
                log_loss_hc = float(-(y_test_hc * np.log(p_test_hc + 1e-12) + (1 - y_test_hc) * np.log(1 - p_test_hc + 1e-12)).mean())

                all_results[model]["accuracy_high_conf"].append(accuracy_hc)
                all_results[model]["log_loss_high_conf"].append(log_loss_hc)
                all_results[model]["n_test_high_conf"].append(n_high_conf)
            else:
                all_results[model]["accuracy_high_conf"].append(np.nan)
                all_results[model]["log_loss_high_conf"].append(np.nan)
                all_results[model]["n_test_high_conf"].append(0)

        # Progress summary
        if (bootstrap_iter + 1) % 10 == 0 or bootstrap_iter == 0:
            print(f"  Completed {bootstrap_iter + 1} iterations")

    # Compute summary statistics
    summary_records: List[Dict[str, object]] = []

    # Print ALL test data results
    print("\n" + "=" * 120)
    print("POOLED MODEL VALIDATION - ALL TEST DATA (per-model breakdown)")
    print("=" * 120)
    print(f"{'Model':<20} {'Acc Mean':>8} {'Acc SE':>8} {'Acc 95% CI':<18} {'LL Mean':>8} {'LL SE':>8} {'N_test':>10}")
    print("-" * 120)

    for model in MODEL_NAMES:
        acc = np.array(all_results[model]["accuracy"])
        ll = np.array(all_results[model]["log_loss"])
        n_test = np.array(all_results[model]["n_test"])
        n_train = np.array(all_results[model]["n_train"])

        acc_hc = np.array(all_results[model]["accuracy_high_conf"])
        ll_hc = np.array(all_results[model]["log_loss_high_conf"])
        n_test_hc = np.array(all_results[model]["n_test_high_conf"])
        
        valid_hc = ~np.isnan(acc_hc)
        acc_hc_valid = acc_hc[valid_hc]
        ll_hc_valid = ll_hc[valid_hc]
        n_test_hc_valid = n_test_hc[valid_hc]

        if len(acc) == 0:
            print(f"{model:<20} {'N/A':>8} {'N/A':>8} {'N/A':<18} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            continue

        acc_mean = acc.mean()
        acc_se = acc.std(ddof=1)
        acc_lo = np.percentile(acc, 2.5)
        acc_hi = np.percentile(acc, 97.5)
        acc_clt_lo = acc_mean - 1.96 * acc_se
        acc_clt_hi = acc_mean + 1.96 * acc_se

        ll_mean = ll.mean()
        ll_se = ll.std(ddof=1)
        ll_lo = np.percentile(ll, 2.5)
        ll_hi = np.percentile(ll, 97.5)
        ll_clt_lo = ll_mean - 1.96 * ll_se
        ll_clt_hi = ll_mean + 1.96 * ll_se

        print(
            f"{model:<20} "
            f"{acc_mean:>8.4f} {acc_se:>8.4f} ({acc_lo:.4f}, {acc_hi:.4f}) "
            f"{ll_mean:>8.4f} {ll_se:>8.4f} {int(n_test.mean()):>10}"
        )

        record: Dict[str, object] = {
            "model": model,
            "n_bootstrap": len(acc),
            "accuracy_mean": acc_mean,
            "accuracy_se": acc_se,
            "accuracy_ci_lo_quantile": acc_lo,
            "accuracy_ci_hi_quantile": acc_hi,
            "accuracy_ci_lo_clt": acc_clt_lo,
            "accuracy_ci_hi_clt": acc_clt_hi,
            "log_loss_mean": ll_mean,
            "log_loss_se": ll_se,
            "log_loss_ci_lo_quantile": ll_lo,
            "log_loss_ci_hi_quantile": ll_hi,
            "log_loss_ci_lo_clt": ll_clt_lo,
            "log_loss_ci_hi_clt": ll_clt_hi,
            "n_test_mean": n_test.mean(),
            "n_train_mean": n_train.mean(),
        }

        if len(acc_hc_valid) > 0:
            acc_hc_mean = acc_hc_valid.mean()
            acc_hc_se = acc_hc_valid.std(ddof=1) if len(acc_hc_valid) > 1 else 0.0
            acc_hc_lo = np.percentile(acc_hc_valid, 2.5)
            acc_hc_hi = np.percentile(acc_hc_valid, 97.5)

            ll_hc_mean = ll_hc_valid.mean()
            ll_hc_se = ll_hc_valid.std(ddof=1) if len(ll_hc_valid) > 1 else 0.0
            ll_hc_lo = np.percentile(ll_hc_valid, 2.5)
            ll_hc_hi = np.percentile(ll_hc_valid, 97.5)

            record.update({
                "accuracy_high_conf_mean": acc_hc_mean,
                "accuracy_high_conf_se": acc_hc_se,
                "accuracy_high_conf_ci_lo": acc_hc_lo,
                "accuracy_high_conf_ci_hi": acc_hc_hi,
                "log_loss_high_conf_mean": ll_hc_mean,
                "log_loss_high_conf_se": ll_hc_se,
                "log_loss_high_conf_ci_lo": ll_hc_lo,
                "log_loss_high_conf_ci_hi": ll_hc_hi,
                "n_test_high_conf_mean": n_test_hc_valid.mean(),
            })
        else:
            record.update({
                "accuracy_high_conf_mean": np.nan,
                "accuracy_high_conf_se": np.nan,
                "accuracy_high_conf_ci_lo": np.nan,
                "accuracy_high_conf_ci_hi": np.nan,
                "log_loss_high_conf_mean": np.nan,
                "log_loss_high_conf_se": np.nan,
                "log_loss_high_conf_ci_lo": np.nan,
                "log_loss_high_conf_ci_hi": np.nan,
                "n_test_high_conf_mean": 0,
            })

        summary_records.append(record)

    # Print HIGH CONFIDENCE results
    print("\n" + "=" * 120)
    print(f"POOLED MODEL - HIGH CONFIDENCE (model-specific values with >= {min_high_conf_comparisons} train comparisons)")
    print("=" * 120)
    print(f"{'Model':<20} {'Acc Mean':>8} {'Acc SE':>8} {'Acc 95% CI':<18} {'LL Mean':>8} {'LL SE':>8} {'N_test':>10}")
    print("-" * 120)

    for model in MODEL_NAMES:
        acc_hc = np.array(all_results[model]["accuracy_high_conf"])
        ll_hc = np.array(all_results[model]["log_loss_high_conf"])
        n_test_hc = np.array(all_results[model]["n_test_high_conf"])
        
        valid_hc = ~np.isnan(acc_hc)
        acc_hc_valid = acc_hc[valid_hc]
        ll_hc_valid = ll_hc[valid_hc]
        n_test_hc_valid = n_test_hc[valid_hc]

        if len(acc_hc_valid) == 0:
            print(f"{model:<20} {'N/A':>8} {'N/A':>8} {'N/A':<18} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            continue

        acc_hc_mean = acc_hc_valid.mean()
        acc_hc_se = acc_hc_valid.std(ddof=1) if len(acc_hc_valid) > 1 else 0.0
        acc_hc_lo = np.percentile(acc_hc_valid, 2.5)
        acc_hc_hi = np.percentile(acc_hc_valid, 97.5)

        ll_hc_mean = ll_hc_valid.mean()
        ll_hc_se = ll_hc_valid.std(ddof=1) if len(ll_hc_valid) > 1 else 0.0
        ll_hc_lo = np.percentile(ll_hc_valid, 2.5)
        ll_hc_hi = np.percentile(ll_hc_valid, 97.5)

        print(
            f"{model:<20} "
            f"{acc_hc_mean:>8.4f} {acc_hc_se:>8.4f} ({acc_hc_lo:.4f}, {acc_hc_hi:.4f}) "
            f"{ll_hc_mean:>8.4f} {ll_hc_se:>8.4f} {int(n_test_hc_valid.mean()):>10}"
        )

    print("-" * 120)

    # Overall summary
    print("\n" + "=" * 120)
    print("OVERALL SUMMARY (pooled across models)")
    print("=" * 120)

    all_acc = np.concatenate([all_results[m]["accuracy"] for m in MODEL_NAMES])
    all_ll = np.concatenate([all_results[m]["log_loss"] for m in MODEL_NAMES])
    all_acc_hc = np.concatenate([all_results[m]["accuracy_high_conf"] for m in MODEL_NAMES])
    all_ll_hc = np.concatenate([all_results[m]["log_loss_high_conf"] for m in MODEL_NAMES])

    if len(all_acc) > 0:
        print(f"  All test data:        Acc={all_acc.mean():.4f} (SE={all_acc.std(ddof=1):.4f}), LL={all_ll.mean():.4f} (SE={all_ll.std(ddof=1):.4f})")

    valid_hc = ~np.isnan(all_acc_hc)
    if valid_hc.sum() > 0:
        acc_hc_v = all_acc_hc[valid_hc]
        ll_hc_v = all_ll_hc[valid_hc]
        print(f"  High confidence:      Acc={acc_hc_v.mean():.4f} (SE={acc_hc_v.std(ddof=1):.4f}), LL={ll_hc_v.mean():.4f} (SE={ll_hc_v.std(ddof=1):.4f})")

    print("\nNote: SE = bootstrap standard error (std of bootstrap samples)")
    print("      95% CI = quantile-based (2.5%, 97.5% percentiles)")
    print(f"      High confidence = test pairs where both values had >= {min_high_conf_comparisons} train comparisons IN THAT MODEL")
    print("      This validation uses ONE pooled BT model fitted on all models' training data,")
    print("      then evaluates per-model test accuracy to assess universal value preferences.")

    # Save to CSV if requested
    if validation_output_csv and summary_records:
        summary_df = pd.DataFrame.from_records(summary_records)
        summary_df.to_csv(validation_output_csv, index=False)
        print(f"\nValidation summary saved to {validation_output_csv}")

    print("\nDone.")


def run_validation(
    parquet_glob: str,
    n_bootstrap: int,
    test_frac: float,
    min_value_comparisons: int,
    min_high_conf_comparisons: int,
    l2_reg: float,
    max_iter: int,
    tol: float,
    seed: int | None,
    validation_output_csv: str | None,
    aggregation_map: Dict[str, str] | None = None,
) -> None:
    """
    Run bootstrap validation: train on (1-test_frac) of data, evaluate on test_frac.
    
    Fits SEPARATE models for each AI system.
    
    Also computes "high confidence" accuracy for test comparisons where both values
    had at least min_high_conf_comparisons in the training set.
    """
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed set to {seed}")

    con = duckdb.connect()

    print("Building global value mapping from parquet data...")
    value_to_idx, idx_to_value = build_value_mapping(con, parquet_glob, aggregation_map)
    num_values_global = len(value_to_idx)
    print(f"Found {num_values_global} distinct values.")

    # Store results: {model: {"accuracy": [...], "log_loss": [...], "n_test": [...]}}
    all_results: Dict[str, Dict[str, List[float]]] = {
        model: {
            "accuracy": [], "log_loss": [], "n_test": [], "n_train": [], "n_values": [],
            "accuracy_high_conf": [], "log_loss_high_conf": [], "n_test_high_conf": [],
        }
        for model in MODEL_NAMES
    }

    print(f"\nRunning {n_bootstrap} bootstrap iterations (test_frac={test_frac})...")
    print(f"High-confidence threshold: {min_high_conf_comparisons} train comparisons per value")
    print("=" * 70)

    for bootstrap_iter in range(n_bootstrap):
        print(f"\n--- Bootstrap iteration {bootstrap_iter + 1}/{n_bootstrap} ---")

        for model in MODEL_NAMES:
            # Load all data for this model
            v1_idx, v2_idx, y, bias = load_model_data_raw(con, parquet_glob, model, value_to_idx, aggregation_map)
            n_total = len(y)

            # Random train/test split
            perm = np.random.permutation(n_total)
            n_test = int(n_total * test_frac)
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]

            # Train data
            v1_train = v1_idx[train_idx]
            v2_train = v2_idx[train_idx]
            y_train = y[train_idx]
            bias_train = bias[train_idx]

            # Apply min_value_comparisons filter on TRAIN set
            counts_train = np.bincount(
                np.concatenate([v1_train, v2_train]),
                minlength=num_values_global,
            )
            fitted_values_mask = counts_train >= min_value_comparisons
            fitted_values = np.where(fitted_values_mask)[0]

            # Filter train rows to only include fitted values
            train_row_mask = fitted_values_mask[v1_train] & fitted_values_mask[v2_train]
            v1_train = v1_train[train_row_mask]
            v2_train = v2_train[train_row_mask]
            y_train = y_train[train_row_mask]
            bias_train = bias_train[train_row_mask]

            if len(v1_train) == 0:
                print(f"  {model}: No train data after filtering, skipping.")
                continue

            # Remap to local indices for fitting
            local_index_for_global = {int(g): int(i) for i, g in enumerate(fitted_values)}
            v1_train_local = np.array([local_index_for_global[g] for g in v1_train], dtype=np.int32)
            v2_train_local = np.array([local_index_for_global[g] for g in v2_train], dtype=np.int32)

            # Fit model on train
            theta_local, alpha, beta, opt_info = fit_bt_model(
                num_values=len(fitted_values),
                v1_idx=v1_train_local,
                v2_idx=v2_train_local,
                y=y_train,
                bias=bias_train,
                l2_reg=l2_reg,
                max_iter=max_iter,
                tol=tol,
            )

            # Expand theta to global indices
            theta_global = np.full(num_values_global, np.nan, dtype=np.float64)
            theta_global[fitted_values] = theta_local

            # Test data: filter to only include comparisons where BOTH values were fitted
            v1_test = v1_idx[test_idx]
            v2_test = v2_idx[test_idx]
            y_test = y[test_idx]
            bias_test = bias[test_idx]

            test_row_mask = fitted_values_mask[v1_test] & fitted_values_mask[v2_test]
            v1_test = v1_test[test_row_mask]
            v2_test = v2_test[test_row_mask]
            y_test = y_test[test_row_mask]
            bias_test = bias_test[test_row_mask]

            if len(y_test) == 0:
                print(f"  {model}: No test data for fitted values, skipping.")
                continue

            # Predict on test set
            eta_test = alpha + (theta_global[v1_test] - theta_global[v2_test]) + beta * bias_test
            eta_test = np.clip(eta_test, -30.0, 30.0)
            p_test = 1.0 / (1.0 + np.exp(-eta_test))

            # Compute metrics (all test data)
            pred = (p_test > 0.5).astype(float)
            accuracy = float((pred == y_test).mean())
            log_loss = float(-(y_test * np.log(p_test + 1e-12) + (1 - y_test) * np.log(1 - p_test + 1e-12)).mean())

            all_results[model]["accuracy"].append(accuracy)
            all_results[model]["log_loss"].append(log_loss)
            all_results[model]["n_test"].append(len(y_test))
            all_results[model]["n_train"].append(len(y_train))
            all_results[model]["n_values"].append(len(fitted_values))

            # High-confidence metrics: only test pairs where BOTH values had >= min_high_conf_comparisons in train
            high_conf_mask = (counts_train[v1_test] >= min_high_conf_comparisons) & (counts_train[v2_test] >= min_high_conf_comparisons)
            n_high_conf = high_conf_mask.sum()

            if n_high_conf > 0:
                pred_hc = pred[high_conf_mask]
                y_test_hc = y_test[high_conf_mask]
                p_test_hc = p_test[high_conf_mask]

                accuracy_hc = float((pred_hc == y_test_hc).mean())
                log_loss_hc = float(-(y_test_hc * np.log(p_test_hc + 1e-12) + (1 - y_test_hc) * np.log(1 - p_test_hc + 1e-12)).mean())

                all_results[model]["accuracy_high_conf"].append(accuracy_hc)
                all_results[model]["log_loss_high_conf"].append(log_loss_hc)
                all_results[model]["n_test_high_conf"].append(n_high_conf)
            else:
                # No high-confidence test samples
                all_results[model]["accuracy_high_conf"].append(np.nan)
                all_results[model]["log_loss_high_conf"].append(np.nan)
                all_results[model]["n_test_high_conf"].append(0)

        # Progress summary for this iteration
        if (bootstrap_iter + 1) % 10 == 0 or bootstrap_iter == 0:
            print(f"  Completed {bootstrap_iter + 1} iterations")

    # Compute summary statistics
    summary_records: List[Dict[str, object]] = []

    # Print ALL test data results
    print("\n" + "=" * 120)
    print("VALIDATION RESULTS - ALL TEST DATA")
    print("=" * 120)
    print(f"{'Model':<20} {'Acc Mean':>8} {'Acc SE':>8} {'Acc 95% CI':<18} {'LL Mean':>8} {'LL SE':>8} {'N_test':>10}")
    print("-" * 120)

    for model in MODEL_NAMES:
        acc = np.array(all_results[model]["accuracy"])
        ll = np.array(all_results[model]["log_loss"])
        n_test = np.array(all_results[model]["n_test"])
        n_train = np.array(all_results[model]["n_train"])
        n_values = np.array(all_results[model]["n_values"])

        # High confidence arrays (filter out NaN)
        acc_hc = np.array(all_results[model]["accuracy_high_conf"])
        ll_hc = np.array(all_results[model]["log_loss_high_conf"])
        n_test_hc = np.array(all_results[model]["n_test_high_conf"])
        
        # Filter out NaN values for high-conf stats
        valid_hc = ~np.isnan(acc_hc)
        acc_hc_valid = acc_hc[valid_hc]
        ll_hc_valid = ll_hc[valid_hc]
        n_test_hc_valid = n_test_hc[valid_hc]

        if len(acc) == 0:
            print(f"{model:<20} {'N/A':>8} {'N/A':>8} {'N/A':<18} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            continue

        # Bootstrap SE = std of bootstrap samples (this IS the SE estimate)
        acc_mean = acc.mean()
        acc_se = acc.std(ddof=1)  # Bootstrap SE
        acc_lo = np.percentile(acc, 2.5)
        acc_hi = np.percentile(acc, 97.5)
        acc_clt_lo = acc_mean - 1.96 * acc_se
        acc_clt_hi = acc_mean + 1.96 * acc_se

        ll_mean = ll.mean()
        ll_se = ll.std(ddof=1)
        ll_lo = np.percentile(ll, 2.5)
        ll_hi = np.percentile(ll, 97.5)
        ll_clt_lo = ll_mean - 1.96 * ll_se
        ll_clt_hi = ll_mean + 1.96 * ll_se

        print(
            f"{model:<20} "
            f"{acc_mean:>8.4f} {acc_se:>8.4f} ({acc_lo:.4f}, {acc_hi:.4f}) "
            f"{ll_mean:>8.4f} {ll_se:>8.4f} {int(n_test.mean()):>10}"
        )

        # Build summary record
        record: Dict[str, object] = {
            "model": model,
            "n_bootstrap": len(acc),
            "accuracy_mean": acc_mean,
            "accuracy_se": acc_se,
            "accuracy_ci_lo_quantile": acc_lo,
            "accuracy_ci_hi_quantile": acc_hi,
            "accuracy_ci_lo_clt": acc_clt_lo,
            "accuracy_ci_hi_clt": acc_clt_hi,
            "log_loss_mean": ll_mean,
            "log_loss_se": ll_se,
            "log_loss_ci_lo_quantile": ll_lo,
            "log_loss_ci_hi_quantile": ll_hi,
            "log_loss_ci_lo_clt": ll_clt_lo,
            "log_loss_ci_hi_clt": ll_clt_hi,
            "n_test_mean": n_test.mean(),
            "n_train_mean": n_train.mean(),
            "n_values_mean": n_values.mean(),
        }

        # Add high-confidence metrics if available
        if len(acc_hc_valid) > 0:
            acc_hc_mean = acc_hc_valid.mean()
            acc_hc_se = acc_hc_valid.std(ddof=1) if len(acc_hc_valid) > 1 else 0.0
            acc_hc_lo = np.percentile(acc_hc_valid, 2.5)
            acc_hc_hi = np.percentile(acc_hc_valid, 97.5)

            ll_hc_mean = ll_hc_valid.mean()
            ll_hc_se = ll_hc_valid.std(ddof=1) if len(ll_hc_valid) > 1 else 0.0
            ll_hc_lo = np.percentile(ll_hc_valid, 2.5)
            ll_hc_hi = np.percentile(ll_hc_valid, 97.5)

            record.update({
                "accuracy_high_conf_mean": acc_hc_mean,
                "accuracy_high_conf_se": acc_hc_se,
                "accuracy_high_conf_ci_lo": acc_hc_lo,
                "accuracy_high_conf_ci_hi": acc_hc_hi,
                "log_loss_high_conf_mean": ll_hc_mean,
                "log_loss_high_conf_se": ll_hc_se,
                "log_loss_high_conf_ci_lo": ll_hc_lo,
                "log_loss_high_conf_ci_hi": ll_hc_hi,
                "n_test_high_conf_mean": n_test_hc_valid.mean(),
            })
        else:
            record.update({
                "accuracy_high_conf_mean": np.nan,
                "accuracy_high_conf_se": np.nan,
                "accuracy_high_conf_ci_lo": np.nan,
                "accuracy_high_conf_ci_hi": np.nan,
                "log_loss_high_conf_mean": np.nan,
                "log_loss_high_conf_se": np.nan,
                "log_loss_high_conf_ci_lo": np.nan,
                "log_loss_high_conf_ci_hi": np.nan,
                "n_test_high_conf_mean": 0,
            })

        summary_records.append(record)

    # Print HIGH CONFIDENCE results
    print("\n" + "=" * 120)
    print(f"VALIDATION RESULTS - HIGH CONFIDENCE (values with >= {min_high_conf_comparisons} train comparisons)")
    print("=" * 120)
    print(f"{'Model':<20} {'Acc Mean':>8} {'Acc SE':>8} {'Acc 95% CI':<18} {'LL Mean':>8} {'LL SE':>8} {'N_test':>10}")
    print("-" * 120)

    for model in MODEL_NAMES:
        acc_hc = np.array(all_results[model]["accuracy_high_conf"])
        ll_hc = np.array(all_results[model]["log_loss_high_conf"])
        n_test_hc = np.array(all_results[model]["n_test_high_conf"])
        
        valid_hc = ~np.isnan(acc_hc)
        acc_hc_valid = acc_hc[valid_hc]
        ll_hc_valid = ll_hc[valid_hc]
        n_test_hc_valid = n_test_hc[valid_hc]

        if len(acc_hc_valid) == 0:
            print(f"{model:<20} {'N/A':>8} {'N/A':>8} {'N/A':<18} {'N/A':>8} {'N/A':>8} {'N/A':>10}")
            continue

        acc_hc_mean = acc_hc_valid.mean()
        acc_hc_se = acc_hc_valid.std(ddof=1) if len(acc_hc_valid) > 1 else 0.0
        acc_hc_lo = np.percentile(acc_hc_valid, 2.5)
        acc_hc_hi = np.percentile(acc_hc_valid, 97.5)

        ll_hc_mean = ll_hc_valid.mean()
        ll_hc_se = ll_hc_valid.std(ddof=1) if len(ll_hc_valid) > 1 else 0.0
        ll_hc_lo = np.percentile(ll_hc_valid, 2.5)
        ll_hc_hi = np.percentile(ll_hc_valid, 97.5)

        print(
            f"{model:<20} "
            f"{acc_hc_mean:>8.4f} {acc_hc_se:>8.4f} ({acc_hc_lo:.4f}, {acc_hc_hi:.4f}) "
            f"{ll_hc_mean:>8.4f} {ll_hc_se:>8.4f} {int(n_test_hc_valid.mean()):>10}"
        )

    print("-" * 120)

    # Overall summary
    print("\n" + "=" * 120)
    print("OVERALL SUMMARY (pooled across models)")
    print("=" * 120)

    all_acc = np.concatenate([all_results[m]["accuracy"] for m in MODEL_NAMES])
    all_ll = np.concatenate([all_results[m]["log_loss"] for m in MODEL_NAMES])
    all_acc_hc = np.concatenate([all_results[m]["accuracy_high_conf"] for m in MODEL_NAMES])
    all_ll_hc = np.concatenate([all_results[m]["log_loss_high_conf"] for m in MODEL_NAMES])

    if len(all_acc) > 0:
        print(f"  All test data:        Acc={all_acc.mean():.4f} (SE={all_acc.std(ddof=1):.4f}), LL={all_ll.mean():.4f} (SE={all_ll.std(ddof=1):.4f})")

    valid_hc = ~np.isnan(all_acc_hc)
    if valid_hc.sum() > 0:
        acc_hc_v = all_acc_hc[valid_hc]
        ll_hc_v = all_ll_hc[valid_hc]
        print(f"  High confidence:      Acc={acc_hc_v.mean():.4f} (SE={acc_hc_v.std(ddof=1):.4f}), LL={ll_hc_v.mean():.4f} (SE={ll_hc_v.std(ddof=1):.4f})")

    print("\nNote: SE = bootstrap standard error (std of bootstrap samples)")
    print("      95% CI = quantile-based (2.5%, 97.5% percentiles)")
    print(f"      High confidence = test pairs where both values had >= {min_high_conf_comparisons} train comparisons")

    # Save to CSV if requested
    if validation_output_csv and summary_records:
        summary_df = pd.DataFrame.from_records(summary_records)
        summary_df.to_csv(validation_output_csv, index=False)
        print(f"\nValidation summary saved to {validation_output_csv}")

    print("\nDone.")


def join_scores_with_metadata(
    scores_df: pd.DataFrame,
    metadata_path: str,
) -> pd.DataFrame:
    """Left-join BT scores with value metadata."""
    meta_df = pd.read_csv(metadata_path)

    # Expect columns: value_name in scores, val in metadata.
    if "value_name" not in scores_df.columns:
        raise ValueError("Scores DataFrame must contain a 'value_name' column.")
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

    return merged


def run_pipeline(
    parquet_glob: str,
    output_csv: str,
    sample_frac: float,
    max_rows: int | None,
    min_value_comparisons: int,
    l2_reg: float,
    max_iter: int,
    tol: float,
    seed: int | None,
    aggregation_map: Dict[str, str] | None = None,
    metadata_csv: str | None = None,
) -> None:
    """
    Orchestrate data loading, per-model fitting, and result aggregation.
    
    If metadata_csv is provided, also generates a version with metadata joined.
    """
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError(f"sample_frac must be in (0, 1], got {sample_frac}.")

    # Set numpy random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed set to {seed}")

    con = duckdb.connect()  # in-memory is fine; we only query parquet_scan

    print("Building global value mapping from parquet data...")
    value_to_idx, idx_to_value = build_value_mapping(con, parquet_glob, aggregation_map)
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
            seed=seed,
            aggregation_map=aggregation_map,
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

        theta_local, alpha, beta, opt_info = fit_bt_model(
            num_values=len(active_values),
            v1_idx=v1_local,
            v2_idx=v2_local,
            y=y,
            bias=bias,
            l2_reg=l2_reg,
            max_iter=max_iter,
            tol=tol,
        )
        
        # Report optimization result
        status = "converged" if opt_info["success"] else "did NOT converge"
        print(f"  Optimization {status} in {opt_info['n_iter']} iterations (nll={opt_info['final_nll']:.4f})")

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
    
    # Also create version with metadata joined if requested
    if metadata_csv:
        import os
        base, ext = os.path.splitext(output_csv)
        with_meta_csv = f"{base}_with_meta{ext}"
        
        print(f"\nJoining with metadata from {metadata_csv} ...")
        merged_df = join_scores_with_metadata(result_df, metadata_csv)
        
        print(f"Writing merged results to {with_meta_csv} ...")
        merged_df.to_csv(with_meta_csv, index=False)
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
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum number of L-BFGS-B iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="Convergence tolerance (ftol) for L-BFGS-B optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not set, results may vary between runs.",
    )
    # Validation mode arguments
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run bootstrap validation mode (separate models per AI) instead of fitting and saving scores.",
    )
    parser.add_argument(
        "--validate-pooled",
        action="store_true",
        help="Run bootstrap validation with ONE pooled model across all AIs, reporting per-AI test accuracy.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=50,
        help="Number of bootstrap iterations for validation mode.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of data to hold out for testing in validation mode.",
    )
    parser.add_argument(
        "--min-high-conf-comparisons",
        type=int,
        default=30,
        help="Minimum train comparisons per value for 'high confidence' accuracy (default 30).",
    )
    parser.add_argument(
        "--validation-output-csv",
        type=str,
        default=None,
        help="Output CSV path for validation summary results (optional).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Use aggregated/merged value names from --aggregate-csv instead of original values.",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=str,
        default="merged_label_values.csv",
        help="CSV file with value aggregation mapping (must have 'value_name' and 'merged_value_names' columns).",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default="labeled_topk_values.csv",
        help="CSV file with value metadata to join with scores (must have 'val' column). "
        "If provided, creates an additional output file with '_with_meta' suffix containing joined data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.validate and args.validate_pooled:
        raise ValueError("Cannot specify both --validate and --validate-pooled. Choose one validation mode.")

    # Load aggregation mapping if requested
    aggregation_map: Dict[str, str] | None = None
    if args.aggregate:
        aggregation_map = load_aggregation_mapping(args.aggregate_csv)

    if args.validate_pooled:
        # Pooled validation mode: one BT model on all AIs' data, per-AI test accuracy
        run_validation_pooled(
            parquet_glob=args.parquet_glob,
            n_bootstrap=args.n_bootstrap,
            test_frac=args.test_frac,
            min_value_comparisons=args.min_value_comparisons,
            min_high_conf_comparisons=args.min_high_conf_comparisons,
            l2_reg=args.l2_reg,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
            validation_output_csv=args.validation_output_csv,
            aggregation_map=aggregation_map,
        )
    elif args.validate:
        # Validation mode: separate models per AI, bootstrap held-out accuracy
        run_validation(
            parquet_glob=args.parquet_glob,
            n_bootstrap=args.n_bootstrap,
            test_frac=args.test_frac,
            min_value_comparisons=args.min_value_comparisons,
            min_high_conf_comparisons=args.min_high_conf_comparisons,
            l2_reg=args.l2_reg,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
            validation_output_csv=args.validation_output_csv,
            aggregation_map=aggregation_map,
        )
    else:
        # Normal mode: fit and save scores
        max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
        run_pipeline(
            parquet_glob=args.parquet_glob,
            output_csv=args.output_csv,
            sample_frac=args.sample_frac,
            max_rows=max_rows,
            min_value_comparisons=args.min_value_comparisons,
            l2_reg=args.l2_reg,
            max_iter=args.max_iter,
            tol=args.tol,
            seed=args.seed,
            aggregation_map=aggregation_map,
            metadata_csv=args.metadata_csv,
        )


if __name__ == "__main__":
    main()


