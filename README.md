Commands ran so far
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

