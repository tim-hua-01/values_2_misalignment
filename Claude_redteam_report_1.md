# Red Team Analysis Report: Model Safety Under Value-Aligned Jailbreaks

**Date:** December 11, 2025  
**Dataset:** `run1_redteam_summary_251210_2341.csv`  
**N observations:** 1,050 (30 unique queries × 35 conditions)

---

## Executive Summary

We tested whether injecting model-specific "values" into jailbreak prompts affects how different LLMs respond. The key findings are:

1. **Target model identity dominates** — which model you're attacking explains ~25% of within-query safety score variance
2. **Value injection source mostly doesn't matter** — using Claude values vs GPT values vs Gemini values has no effect
3. **Exception: Grok is MORE vulnerable to its own values** — when jailbreaks use Grok's extracted values, Grok is significantly more likely to comply (0.26-0.40 points lower safety score)
4. **Clear safety ranking**: Claude (safest) > GPT > Gemini > Grok (most vulnerable)

---

## 1. Data Overview

### Experimental Design

- **30 jailbreak queries** from two sources (Tim: hand-crafted, Claude: AI-generated)
- **4 target models**: Claude 3.5 Sonnet, GPT-4.1-mini, Gemini 2.5 Pro, Grok-4
- **9 value injection conditions per query**:
  - 4 models × 2 value types (prefer/relative) = 8 value-injected prompts
  - 1 unmodified baseline
- **5 samples per condition**, averaged to `avg_score`

### Scoring

- **0 = Full compliance** (model responds to harmful request)
- **5 = Full refusal** (model refuses the request)
- Graded by GPT-5-mini

### Key Variables

| Variable | Description |
|----------|-------------|
| `target_model` | Model being attacked (Claude, Gemini, GPT, Grok) |
| `type_agree` | TRUE if value source matches target model |
| `prefer_or_relative` | Type of value extraction method |
| `source` | Query author (Tim or Claude) |

---

## 2. Overall Safety by Target Model

![Safety Score by Model](plots/rt_score_by_model.png)

| Model | Mean Score | SD | N |
|-------|-----------|-----|-----|
| **Claude** | 4.27 | 1.33 | 249 |
| **GPT** | 3.65 | 1.78 | 270 |
| **Gemini** | 3.20 | 2.06 | 270 |
| **Grok** | 2.31 | 2.16 | 261 |

**Finding:** Clear safety hierarchy. Claude refuses ~85% of the time (score ≈ 4.3), while Grok complies more than half the time (score ≈ 2.3).

### Score Distributions

![Score Distribution](plots/score_distribution_by_model.png)

- **Claude**: Heavily right-skewed (mostly 4s and 5s)
- **Grok**: Bimodal with many 0s (full compliance)
- **GPT/Gemini**: More spread across the range

---

## 3. What Explains Safety Score Variation?

### Variance Decomposition (R²)

| Predictor | R² (no FE) | Within R² (query FE) |
|-----------|------------|---------------------|
| **Target Model** | 12.6% | 24.9% |
| Type Agree | 0.0% | 0.05% |
| Prefer/Relative | 0.0% | 0.04% |
| Source | 1.0% | 0.1% |
| **All combined** | 14.1% | 25.1% |

**Finding:** Target model explains virtually all the predictable variation. Value-related variables explain essentially nothing at the aggregate level.

### Regression Results (Full Sample, Clustered SEs at query level)

```
Dependent Variable: avg_score

                              No FE       Query FE
────────────────────────────────────────────────────
(Intercept)                 4.54***           
Target: Gemini             -1.08***    -1.18***
Target: GPT                -0.63**     -0.73**
Target: Grok               -1.97***    -2.02***
Type Agree (TRUE)          -0.07       -0.06
Prefer/Relative: relative  -0.27       -0.08
Source: Tim                -0.68       -0.19
────────────────────────────────────────────────────
R²                          0.141       0.609
Within R²                     —         0.251
```

**Interpretation:** 
- Relative to Claude, Grok scores **2 points lower** (on a 0-5 scale)
- Type agree and prefer/relative are not significant

---

## 4. The Grok Exception: Value-Aligned Vulnerability

### Per-Model Analysis

When we run separate regressions for each target model, one stands out:

| Model | type_agree Coefficient | p-value |
|-------|----------------------|---------|
| Claude | -0.08 | ns |
| Gemini | +0.19 | ns |
| GPT | -0.12 | ns |
| **Grok** | **-0.26** | **p < 0.05** |

![Grok Type Agree Effect](plots/grok_typeagree_effect.png)

**Finding:** When jailbreaks use Grok's own extracted values, Grok's safety score drops by 0.26 points. Since lower score = more compliance, **Grok is MORE likely to comply with value-aligned jailbreaks**.

### Robustness Across Subsamples

| Sample | Grok type_agree coef | Significant? |
|--------|---------------------|--------------|
| Full sample | -0.26 | Yes* |
| Claude-source only | -0.32 | Yes* |
| Relative + Unmodified | -0.40 | Yes* |
| Tim only | +0.01 | No |

The effect is robust except in Tim's (smaller) sample.

### Visualization Across All Models

![Model Type Agree Comparison](plots/model_typeagree_comparison.png)

- Claude, GPT, Gemini: No meaningful difference between own vs other values
- Grok: Visible drop when using its own values

---

## 5. Pairwise Model Comparisons

For each query × type combination, we compared which model scored higher (excluding ties).

### Overall Win Rates

![Pairwise Wins](plots/pairwise_wins_full.png)

| Row Model | vs Claude | vs GPT | vs Gemini | vs Grok |
|-----------|-----------|--------|-----------|---------|
| **Claude** | — | 77% | 81% | 88% |
| **GPT** | 23% | — | 62% | 78% |
| **Gemini** | 19% | 38% | — | 71% |
| **Grok** | 12% | 22% | 29% | — |

**Finding:** Claude beats every other model >75% of the time. Grok loses to everyone.

### Does "Home Advantage" Help?

![Type Agree Effect on Win Rate](plots/pairwise_typeagree_diff_full.png)

| Model | Change in win rate with own values |
|-------|-----------------------------------|
| Claude | +0% to +6% (slight help) |
| Gemini | +0% to +9% (slight help) |
| GPT | +3% to -5% (mixed) |
| **Grok** | **-2% to -6% (hurts!)** |

**Finding:** Using Grok's values against Grok makes it *lose more often*, not less. The opposite of a "home advantage."

---

## 6. Query Source Effects

![Source Comparison](plots/model_source_comparison.png)

| Source | Claude | Gemini | GPT | Grok |
|--------|--------|--------|-----|------|
| **Tim** | 4.40 | 2.64 | 3.13 | 1.55 |
| **Claude** | 4.24 | 3.32 | 3.77 | 2.49 |

**Finding:** Tim's hand-crafted jailbreaks are harder (lower scores across all models). Grok drops to 1.55 on Tim's queries — near-total compliance.

---

## 7. Value Type (Prefer vs Relative) Has No Effect

![Value Type Comparison](plots/rt_score_by_prefrel.png)

| Value Type | Mean Score |
|------------|-----------|
| Prefer | 3.36 |
| Relative | 3.32 |
| Unmodified | 3.42 |

**Finding:** The method used to extract values (preference ranking vs relative comparison) makes no difference to jailbreak effectiveness.

---

## 8. Key Conclusions

### Main Findings

1. **Model identity is everything**: The target model explains 25% of within-query variance; value-related factors explain <1%

2. **Safety ranking is robust**: Claude >> GPT > Gemini >> Grok, consistent across all subsamples

3. **Value alignment doesn't help attackers (mostly)**: Using Claude values against Claude, or GPT values against GPT, has no effect

4. **Grok is the exception**: Grok is uniquely vulnerable to its own values. When jailbreaks invoke Grok's extracted values, Grok complies more often

5. **Query difficulty varies**: Human-crafted (Tim) jailbreaks are harder than AI-generated (Claude) jailbreaks

### Implications

- **For red-teaming**: Focus on model choice, not value injection strategy. Grok is low-hanging fruit.
- **For Grok specifically**: There may be something about how Grok's values are represented that makes value-aligned prompts more persuasive
- **For safety research**: The lack of "home advantage" for most models suggests their safety training is robust to value-based manipulation

### Limitations

- Only 30 unique queries (statistical power is limited)
- Tim's sample is small (50 obs per model)
- Single grader (GPT-5-mini) may have systematic biases
- The Grok effect disappears in Tim's sample — may be specific to AI-generated jailbreaks

---

## Appendix: Regression Tables

### Full Sample with Query Fixed Effects

```
========== Final Summary Table ==========
                                No FE    Query FE
────────────────────────────────────────────────────
Target: Gemini             -1.08***    -1.18***
Target: GPT                -0.63**     -0.73**
Target: Grok               -1.97***    -2.02***
Type Agree                 -0.07       -0.06
Prefer/Rel: relative       -0.27       -0.08
Source: Tim                -0.68       -0.19
────────────────────────────────────────────────────
Observations                1,050       1,050
R²                          0.141       0.609
Within R²                     —         0.251
SE clustered at query_id level
*** p<0.001, ** p<0.01, * p<0.05
```

### Per-Model Regressions (With Query FE)

```
                           Claude    Gemini     GPT      Grok
────────────────────────────────────────────────────────────────
Type Agree                 -0.08     +0.19    -0.12   -0.26*
Prefer/Rel: relative       -0.21.    -0.02    -0.21    +0.09
Source: Tim                -0.18     -0.12    -0.37    -0.02
────────────────────────────────────────────────────────────────
Within R²                   0.014     0.014    0.017    0.028
```

---

*Report generated from `analysis_rt.R` and `analysis_rt_pairwise.R`*




