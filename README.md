## Project overview

This repo analyzes how different LLMs trade off between competing **values** (e.g. self‑preservation, sales effectiveness, safety, truthfulness) when answering prompts from the *stress_testing_model_spec* dataset.

The core idea:

- **Input**: pairwise comparisons where each prompt pits `value1` vs `value2`, plus per‑model positions indicating which value the model's response favored.
- **Model**: for each LLM separately, fit a Bradley–Terry–style model (with a nudge/bias term) to estimate a scalar score \\(\\theta_v\\) for every value \\(v\\).
- **Output**: per‑model rankings of values and category‑level summaries (e.g. harm‑related vs self‑preservation vs sales‑effectiveness).

These scores can be treated like **Elo ratings over values** within each model and then used for downstream analysis or visualization.

## Data layout

- **`raw_data/data/all_models-0000x-of-00004.parquet`**
  - Main comparisons table, one row per prompt / value pair.
  - Important columns:
    - **`value1`, `value2`**: the two candidate values being traded off.
    - **`nudge_direction`**: which value the prompt text nudged toward (`"value1"`, `"value2"`, or neutral).
    - **Per‑model position columns** (one pair per model), e.g.:
      - `claude_3_5_sonnet_value1_position`, `claude_3_5_sonnet_value2_position`
      - `gemini_2_5_pro_value1_position`, `gemini_2_5_pro_value2_position`
      - `o3_value1_position`, `o3_value2_position`
    - Positions encode how strongly the model's response aligns with each value; higher = more aligned with that value.

- **`labeled_topk_values.csv`**
  - Metadata table for a curated subset of values.
  - Key columns:
    - **`val`**: value name (matches `value1`/`value2` / `value_name`).
    - **`freq`**: how often this value appears in the dataset.
    - **`harm_related`, `self_preservation`, `safety_relatedness`**: thematic flags.
    - **`locus_of_focus`**: `Self` / `User` / `System`.
    - **`agency_v_subservience`, `truthfulness`, `moral_alignment`**.
    - **`abstraction_level`, `instrumental_v_terminal`, `emotional_valence`**.

- **`processed_data/bt_value_scores_full.csv`**
  - Per‑model Bradley–Terry scores fit on the full dataset.
  - Columns:
    - **`model`**: LLM identifier (e.g. `claude_3_5_sonnet`, `gpt_4o`, `o3`).
    - **`value_index`**: global index for the value.
    - **`value_name`**: value string.
    - **`theta`**: Bradley–Terry score for that value under that model (mean‑zero within each model).
    - **`freq`**: number of comparisons this model saw involving that value.
    - **`rank`**: rank of the value within that model (1 = highest `theta`).

- **`processed_data/bt_value_scores_with_meta.csv`**
  - Left‑join of `bt_value_scores_full.csv` with `labeled_topk_values.csv` on `value_name == val`.
  - Same columns as `bt_value_scores_full.csv` plus the value metadata; many values will have missing metadata, which is expected.

## What the modeling code is doing

- **Goal**: for each model \\(M\\), estimate a scalar \\(\\theta_v^{(M)}\\) for every value \\(v\\) such that
  - \\(\\theta_v^{(M)}\\) is high if the model tends to favor that value across comparisons, after correcting for the textual nudge.
- **Model form (per comparison row \\(i\\))**:
  - Let \\(v_1, v_2\\) be the two values; \\(b_i \\in \\{-1,0,1\\}\\) is the nudge bias towards `value1` / `value2` / neutral.
  - Linear predictor:
    - \\(\\eta_i = \\alpha + (\\theta_{v_1} - \\theta_{v_2}) + \\beta b_i\\)
  - Outcome:
    - \\(p_i = \\sigma(\\eta_i) = 1/(1 + e^{-\\eta_i})\\)
    - \\(y_i = 1\\) if the model favored `value1`, \\(0\\) if it favored `value2`.
  - Regularization:
    - L2 penalty on \\(\\theta\\) to stabilize rare values.
    - Identifiability: enforce \\(\\sum_v \\theta_v = 0\\) per model by mean‑centering after each gradient step.

- Interpretation:
  - Within a model, **differences in \\(\\theta\\)** correspond to differences in **log‑odds of preferring one value over another**.
  - You can rescale \\(\\theta\\) to an **Elo‑like rating**:
    - `elo = 1500 + (400/ln(10)) * theta` per model.
  - Cross‑model comparisons of \\(\\theta\\) are on a common logistic scale, but the spread can differ by model, so interpret small cross‑model differences cautiously.

## Key scripts

- **`bt_rank_values.py`**
  - Reads `raw_data/data/*.parquet` via DuckDB (only the needed columns).
  - Builds a global value index from all `value1`/`value2` strings.
  - For each model in `MODEL_NAMES`, constructs:
    - `v1_idx`, `v2_idx`, binary outcomes `y`, and nudge `bias`.
  - Fits the generalized Bradley–Terry model via L-BFGS-B optimization:
    - Options for sampling (`--sample-frac`), min comparisons per value, and validation.
  - Validation modes:
    - `--validate`: Fits separate BT models per AI system (default validation).
    - `--validate-pooled`: Fits ONE universal BT model on all AIs' training data, then evaluates per-AI test accuracy to test for universal value preferences.
  - Writes per‑model scores to:
    - `processed_data/bt_value_scores_full.csv` (or a user‑specified path).

- **`join_bt_with_metadata.py`**
  - Left‑joins BT scores with `labeled_topk_values.csv` on `value_name == val`.
  - Output is `processed_data/bt_value_scores_with_meta.csv`, ready for:
    - Category‑level analyses (e.g. mean \\(\\theta\\) for harm‑related vs self‑preservation).
    - Visualizations (rank distributions by category, etc.).

## Commands ran so far

```
hf download jifanz/stress_testing_model_spec \
    --repo-type dataset \
    --include "data/all_models*" \
    --local-dir raw_data/ 

uv run python bt_rank_values.py \
    --parquet-glob "raw_data/data/*.parquet" \
    --sample-frac 1.0 \
    --max-rows 0 \
    --seed 42 \
    --output-csv "bt_value_scores_full_new.csv"


uv run python bt_rank_values.py \
    --parquet-glob "raw_data/data/*.parquet" \
    --validate \
    --n-bootstrap 50 \
    --test-frac 0.1 \
    --min-value-comparisons 5 \
    --seed 42 \
    --validation-output-csv "validation_results.csv"

uv run python bt_rank_values.py \
    --parquet-glob "raw_data/data/*.parquet" \
    --sample-frac 1.0 \
    --max-rows 0 \
    --seed 42 \
    --min-value-comparisons 30 \
    --output-csv "bt_value_scores_full_30.csv"

uv run python bt_rank_values.py \
    --parquet-glob "raw_data/data/*.parquet" \
    --validate \
    --n-bootstrap 50 \
    --test-frac 0.1 \
    --min-value-comparisons 5 \
    --min-high-conf-comparisons 30 \
    --seed 42 \
    --validation-output-csv "validation_results_high_conf.csv"

# Pooled validation: test if a universal BT model generalizes to individual AIs
uv run python bt_rank_values.py \
    --parquet-glob "raw_data/data/*.parquet" \
    --validate-pooled \
    --n-bootstrap 50 \
    --test-frac 0.1 \
    --min-value-comparisons 5 \
    --min-high-conf-comparisons 30 \
    --seed 42 \
    --validation-output-csv "validation_results_pooled.csv"


  uv run python run_redteam_grading.py  --prefix run1 --n-samples 5


uv run python bt_rank_values.py \
    --parquet-glob "raw_data/data/*.parquet" \
    --aggregate \
    --sample-frac 1.0 \
    --max-rows 0 \
    --output-csv "processed_data/bt_value_scores_aggregated.csv"

uv run python bt_rank_values.py \
    --parquet-glob "raw_data/data/*.parquet" \
    --aggregate \
    --sample-frac 1.0 \
    --max-rows 0 \
    --validate \
    --n-bootstrap 50 \
    --test-frac 0.1 \
    --min-value-comparisons 5 \
    --min-high-conf-comparisons 30 \
    --seed 422

python petri_base_script.py \
    --target "openai/gpt-4.1-mini" \
    --value1 "copyright respect" \
    --value2 "dramatic craft" \
    --epochs 3 \
    --max-turns 6

python petri_base_script.py \
    --target "openai/gpt-4.1-mini" \
    --value1 "protecting property" \
    --value2 "thorough guidance" \
    --epochs 3 \
    --max-turns 6 --csv conflicting_values_agg.csv --log-name test2.eval


python petri_base_script.py \
    --target "openai/gpt-4.1-mini" \
    --value1 "protecting property" \
    --value2 "thorough guidance" \
    --epochs 1 \
    --max-turns 1 --csv conflicting_values_agg.csv --log-name test3.eval

./run_parallel_petri.sh example_parallel_input.csv

./run_parallel_petri.sh example_parallel_input.csv --output test_petri.csv

./run_parallel_petri.sh petri_values_run_minitest.csv --output petri_values_run_results_minitest.csv

./run_parallel_petri.sh petri_values_run1.csv --output petri_values_run1_results.csv

./run_parallel_petri.sh sample_minirun.csv --output sample_minirun_res.csv --logs-dir minirun

```

```
========================================================================================================================
VALIDATION RESULTS - ALL TEST DATA
========================================================================================================================
Model                Acc Mean   Acc SE Acc 95% CI          LL Mean    LL SE     N_test
------------------------------------------------------------------------------------------------------------------------
claude_3_5_sonnet      0.8563   0.0044 (0.8491, 0.8663)   0.3730   0.0116       3823
claude_3_7_sonnet      0.6805   0.0082 (0.6625, 0.6959)   0.6309   0.0121       3797
claude_opus_3          0.7633   0.0060 (0.7520, 0.7750)   0.5379   0.0128       3858
claude_opus_4          0.6382   0.0073 (0.6243, 0.6535)   0.6709   0.0091       3703
claude_sonnet_4        0.6497   0.0071 (0.6349, 0.6600)   0.6572   0.0111       3731
gemini_2_5_pro         0.7502   0.0061 (0.7394, 0.7588)   0.5584   0.0112       3831
gpt_4_1                0.7012   0.0071 (0.6889, 0.7151)   0.6085   0.0108       3807
gpt_4_1_mini           0.7372   0.0061 (0.7249, 0.7477)   0.5672   0.0108       3826
gpt_4o                 0.6394   0.0082 (0.6254, 0.6546)   0.6703   0.0117       3591
grok_4                 0.8812   0.0046 (0.8736, 0.8879)   0.3293   0.0137       3930
o3                     0.6256   0.0084 (0.6130, 0.6398)   0.6806   0.0084       3695
o4_mini                0.6244   0.0081 (0.6098, 0.6391)   0.6803   0.0090       3779

========================================================================================================================
VALIDATION RESULTS - HIGH CONFIDENCE (values with >= 30 train comparisons)
========================================================================================================================
Model                Acc Mean   Acc SE Acc 95% CI          LL Mean    LL SE     N_test
------------------------------------------------------------------------------------------------------------------------
claude_3_5_sonnet      0.8539   0.0231 (0.8161, 0.9019)   0.3717   0.0569        178
claude_3_7_sonnet      0.6730   0.0390 (0.5867, 0.7334)   0.6389   0.0532        170
claude_opus_3          0.7630   0.0268 (0.7180, 0.8115)   0.5213   0.0542        180
claude_opus_4          0.6857   0.0383 (0.6042, 0.7561)   0.6048   0.0412        141
claude_sonnet_4        0.6979   0.0407 (0.6164, 0.7656)   0.6006   0.0479        151
gemini_2_5_pro         0.7711   0.0307 (0.7161, 0.8204)   0.5150   0.0454        180
gpt_4_1                0.6995   0.0303 (0.6496, 0.7584)   0.6027   0.0438        174
gpt_4_1_mini           0.7172   0.0340 (0.6521, 0.7850)   0.5837   0.0520        184
gpt_4o                 0.6606   0.0471 (0.5831, 0.7536)   0.6440   0.0555        110
grok_4                 0.8696   0.0226 (0.8285, 0.9114)   0.3680   0.0658        214
o3                     0.6318   0.0395 (0.5582, 0.6909)   0.6454   0.0343        122
o4_mini                0.6205   0.0410 (0.5471, 0.7014)   0.6569   0.0424        151
------------------------------------------------------------------------------------------------------------------------

========================================================================================================================
OVERALL SUMMARY (pooled across models)
========================================================================================================================
  All test data:        Acc=0.7123 (SE=0.0845), LL=0.5804 (SE=0.1139)
  High confidence:      Acc=0.7203 (SE=0.0842), LL=0.5628 (SE=0.1086)

Note: SE = bootstrap standard error (std of bootstrap samples)
      95% CI shown is quantile-based (2.5%, 97.5% percentiles)
      CLT-based CI = mean ± 1.96 * SE (also saved in CSV)

### Interpretation of validation modes

- **`--validate`** (separate models): Fits independent BT models for each AI system. This tells us how well each model's preferences can be predicted from its own training data.

- **`--validate-pooled`** (universal model): Fits ONE BT model on pooled training data from all AI systems, then evaluates on each system's test set. This tests whether there exist **universal value preferences** that generalize across different AI systems. Lower accuracy in pooled validation compared to separate validation suggests AI systems have heterogeneous value trade-offs.

Claude notes on interpreting the CE Loss term
Interpretation
Log-Loss	Interpretation
0.693	Random guessing (predicting p=0.5 for everything)
< 0.693	Better than random — your model has learned something
0.5 - 0.6	Decent model
0.3 - 0.5	Good model
< 0.3	Very good calibration
→ 0	Perfect predictions (p=1 when y=1, p=0 when y=0)

```

Get some ideas for what value one can look like with some query examples. 


## Aggregated high confidence validation

```
========================================================================================================================
VALIDATION RESULTS - HIGH CONFIDENCE (values with >= 30 train comparisons)
========================================================================================================================
Model                Acc Mean   Acc SE Acc 95% CI          LL Mean    LL SE     N_test
------------------------------------------------------------------------------------------------------------------------
claude_3_5_sonnet      0.8407   0.0171 (0.8098, 0.8773)   0.3817   0.0371        394
claude_3_7_sonnet      0.6723   0.0231 (0.6268, 0.7126)   0.6182   0.0300        379
claude_opus_3          0.7633   0.0205 (0.7232, 0.8036)   0.5170   0.0333        393
claude_opus_4          0.6711   0.0233 (0.6187, 0.7171)   0.6180   0.0243        361
claude_sonnet_4        0.6787   0.0242 (0.6402, 0.7215)   0.6125   0.0248        370
gemini_2_5_pro         0.7658   0.0191 (0.7302, 0.8033)   0.5202   0.0305        398
gpt_4_1                0.7024   0.0222 (0.6601, 0.7476)   0.5852   0.0263        380
gpt_4_1_mini           0.7243   0.0233 (0.6816, 0.7691)   0.5662   0.0305        395
gpt_4o                 0.6641   0.0251 (0.6097, 0.7074)   0.6280   0.0245        318
grok_4                 0.8682   0.0183 (0.8305, 0.8991)   0.3510   0.0454        428
o3                     0.6367   0.0341 (0.5687, 0.6921)   0.6428   0.0226        331
o4_mini                0.6302   0.0243 (0.5819, 0.6694)   0.6497   0.0239        353

```