################################################################################
# Analysis: Do BT Ranks/Thetas Predict Multi-Turn Auditing Scores?
# Thinking like an economist: Multiple validation metrics
################################################################################

setwd("~/Documents/aisafety_githubs/values_2_misalignment")

# Load packages
library(tidyverse)
library(magrittr)
library(fixest)
library(broom)
library(modelsummary)

# Custom theme and colors from analysis_1.R
myTheme <- theme(plot.title = element_text(size = 15),
                 panel.background = element_rect(fill = '#F2F2ED'),
                 legend.text = element_text(size = 10),
                 plot.subtitle = element_text(size = 12),
                 axis.title = element_text(size = 12),
                 axis.text = element_text(size = 12, colour = 'black'),
                 legend.position = "bottom",
                 legend.background = element_rect(linetype = 3, size = 0.5, color = 'black', fill = 'grey94'),
                 legend.key = element_rect(size = 0.5, linetype = 1, color = 'black'))

nicepurp <- "#A88DBF"
niceblue <- '#38A5E0'
nicegreen <- '#A3DCC0'
custom_colors <- c("#2ECC71", "#A3E635", "#F4D03F", "#F39C12", "#E74C3C", "#C0392B", "#0072B2", "#CC79A7")

# Model color palette
model_colors <- c(
  "gpt41" = "#10A37F",      # OpenAI green
  "gemini25" = "#8E75B2",   # Google Gemini purple
  "claude35" = "#CC9B7A",   # Anthropic beige/tan
  "grok4" = "#000000"       # xAI black
)
################################################################################
# 1. LOAD AND COMBINE DATA
################################################################################

#df1 <- read_csv("petri_values_run1_results.csv")
#df2 <- read_csv("petri_values_run_results_minitest.csv")

# Combine datasets
df <- read_csv('petri_values_run1_results_fixed.csv')
cat("Combined dataset:", nrow(df), "rows\n")

# Reshape to long format for analysis
# Each observation: model x value-pair x value-position
df_long <- df %>%
  pivot_longer(
    cols = c(gpt41_v1_score, gpt41_v2_score, gemini25_v1_score, gemini25_v2_score,
             claude35_v1_score, claude35_v2_score, grok4_v1_score, grok4_v2_score),
    names_to = c("model", "value_pos", "metric"),
    names_pattern = "(.*)_(v[12])_(.*)",
    values_to = "score"
  ) %>%
  select(-metric) %>%
  mutate(
    # Get corresponding ranks and thetas based on value position
    rank = case_when(
      model == "gpt41" & value_pos == "v1" ~ value1_gpt_rank,
      model == "gpt41" & value_pos == "v2" ~ value2_gpt_rank,
      model == "gemini25" & value_pos == "v1" ~ value1_gemini_rank,
      model == "gemini25" & value_pos == "v2" ~ value2_gemini_rank,
      model == "claude35" & value_pos == "v1" ~ value1_claude_rank,
      model == "claude35" & value_pos == "v2" ~ value2_claude_rank,
      model == "grok4" & value_pos == "v1" ~ value1_grok_rank,
      model == "grok4" & value_pos == "v2" ~ value2_grok_rank
    ),
    theta = case_when(
      model == "gpt41" & value_pos == "v1" ~ value1_gpt_theta,
      model == "gpt41" & value_pos == "v2" ~ value2_gpt_theta,
      model == "gemini25" & value_pos == "v1" ~ value1_gemini_theta,
      model == "gemini25" & value_pos == "v2" ~ value2_gemini_theta,
      model == "claude35" & value_pos == "v1" ~ value1_claude_theta,
      model == "claude35" & value_pos == "v2" ~ value2_claude_theta,
      model == "grok4" & value_pos == "v1" ~ value1_grok_theta,
      model == "grok4" & value_pos == "v2" ~ value2_grok_theta
    ),
    value_name = if_else(value_pos == "v1", value1_name, value2_name),
    pair_id = paste(value1_name, value2_name, sep = "_vs_")
  ) %>%
  # Keep only relevant columns
  select(pair_id, model, value_pos, value_name, score, rank, theta)

# Create pairwise difference dataset (for each model-pair observation)
# Each row = one value-pair, one model
df_pairs <- df %>%
  mutate(pair_id = row_number()) %>%
  rowwise() %>%
  mutate(
    # GPT
    gpt41_score_diff = gpt41_v1_score - gpt41_v2_score,
    gpt41_rank_v1 = value1_gpt_rank, gpt41_rank_v2 = value2_gpt_rank,
    gpt41_theta_v1 = value1_gpt_theta, gpt41_theta_v2 = value2_gpt_theta,
    # Gemini
    gemini25_score_diff = gemini25_v1_score - gemini25_v2_score,
    gemini25_rank_v1 = value1_gemini_rank, gemini25_rank_v2 = value2_gemini_rank,
    gemini25_theta_v1 = value1_gemini_theta, gemini25_theta_v2 = value2_gemini_theta,
    # Claude
    claude35_score_diff = claude35_v1_score - claude35_v2_score,
    claude35_rank_v1 = value1_claude_rank, claude35_rank_v2 = value2_claude_rank,
    claude35_theta_v1 = value1_claude_theta, claude35_theta_v2 = value2_claude_theta,
    # Grok
    grok4_score_diff = grok4_v1_score - grok4_v2_score,
    grok4_rank_v1 = value1_grok_rank, grok4_rank_v2 = value2_grok_rank,
    grok4_theta_v1 = value1_grok_theta, grok4_theta_v2 = value2_grok_theta
  ) %>%
  ungroup() %>%
  pivot_longer(
    cols = c(gpt41_score_diff, gemini25_score_diff, claude35_score_diff, grok4_score_diff),
    names_to = "model",
    names_pattern = "(.*)_score_diff",
    values_to = "score_diff"
  ) %>%
  mutate(
    rank_v1 = case_when(
      model == "gpt41" ~ gpt41_rank_v1,
      model == "gemini25" ~ gemini25_rank_v1,
      model == "claude35" ~ claude35_rank_v1,
      model == "grok4" ~ grok4_rank_v1
    ),
    rank_v2 = case_when(
      model == "gpt41" ~ gpt41_rank_v2,
      model == "gemini25" ~ gemini25_rank_v2,
      model == "claude35" ~ claude35_rank_v2,
      model == "grok4" ~ grok4_rank_v2
    ),
    theta_v1 = case_when(
      model == "gpt41" ~ gpt41_theta_v1,
      model == "gemini25" ~ gemini25_theta_v1,
      model == "claude35" ~ claude35_theta_v1,
      model == "grok4" ~ grok4_theta_v1
    ),
    theta_v2 = case_when(
      model == "gpt41" ~ gpt41_theta_v2,
      model == "gemini25" ~ gemini25_theta_v2,
      model == "claude35" ~ claude35_theta_v2,
      model == "grok4" ~ grok4_theta_v2
    ),
    rank_diff = rank_v1 - rank_v2,  # negative = v1 ranked higher (lower rank = more valued)
    theta_diff = theta_v1 - theta_v2,  # positive = v1 has higher theta (more valued in BT)
    # Binary outcomes
    v1_wins_score = as.integer(score_diff > 0),
    v1_wins_rank = as.integer(rank_diff < 0),  # v1 ranked higher = lower rank number
    v1_wins_theta = as.integer(theta_diff > 0),
    # Concordance: do rank and score agree?
    concordant = as.integer(v1_wins_rank == v1_wins_score),
    concordant_theta = as.integer(v1_wins_theta == v1_wins_score)
  ) %>%
  # Keep only relevant columns
  select(pair_id, value1_name, value2_name, model,
         score_diff, rank_v1, rank_v2, theta_v1, theta_v2,
         rank_diff, theta_diff, v1_wins_score, v1_wins_rank, v1_wins_theta,
         concordant, concordant_theta)

################################################################################
# 2. METRIC 1: WITHIN-MODEL PAIRWISE CONCORDANCE
# If rank(v1) < rank(v2), is score(v1) > score(v2)?
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 1: Within-Model Pairwise Concordance\n")
cat(strrep("=", 60), "\n")

concordance_by_model <- df_pairs %>%
  group_by(model) %>%
  summarise(
    n_pairs = n(),
    concordant_rank = sum(concordant, na.rm = TRUE),
    concordance_rate_rank = mean(concordant, na.rm = TRUE),
    concordant_theta = sum(concordant_theta, na.rm = TRUE),
    concordance_rate_theta = mean(concordant_theta, na.rm = TRUE),
    .groups = "drop"
  )

print(concordance_by_model) 

# Overall concordance
overall_concordance <- df_pairs %>%
  summarise(
    n_pairs = n(),
    concordance_rate_rank = mean(concordant, na.rm = TRUE),
    concordance_rate_theta = mean(concordant_theta, na.rm = TRUE),
    se_rank = sqrt(concordance_rate_rank * (1 - concordance_rate_rank) / n_pairs),
    se_theta = sqrt(concordance_rate_theta * (1 - concordance_rate_theta) / n_pairs)
  )

cat("\nOverall concordance (rank):", round(overall_concordance$concordance_rate_rank, 3),
    "± ", round(1.96 * overall_concordance$se_rank, 3), "\n")
cat("Overall concordance (theta):", round(overall_concordance$concordance_rate_theta, 3),
    "± ", round(1.96 * overall_concordance$se_theta, 3), "\n")

# Test if significantly different from 0.5 (random)
binom_test_rank <- binom.test(sum(df_pairs$concordant), nrow(df_pairs), p = 0.5)
binom_test_theta <- binom.test(sum(df_pairs$concordant_theta), nrow(df_pairs), p = 0.5)

cat("\nBinomial test (rank) p-value:", format(binom_test_rank$p.value, scientific = TRUE), "\n")
cat("Binomial test (theta) p-value:", format(binom_test_theta$p.value, scientific = TRUE), "\n")

################################################################################
# 3. METRIC 2: CROSS-MODEL ORDERING CONSISTENCY
# For value V, if Claude ranks it higher than Gemini, does Claude score it higher?
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 2: Cross-Model Value Ordering Consistency\n")
cat(strrep("=", 60), "\n")

# Create value-level dataset with average scores and ranks per model
df_values <- df_long %>%
  group_by(model, value_name) %>%
  summarise(
    mean_score = mean(score, na.rm = TRUE),
    mean_rank = mean(rank, na.rm = TRUE),
    mean_theta = mean(theta, na.rm = TRUE),
    n_obs = n(),
    .groups = "drop"
  )

# For each value, check if model ordering by rank matches ordering by score
# Create all model pairs
model_pairs <- combn(c("gpt41", "gemini25", "claude35", "grok4"), 2, simplify = FALSE)

cross_model_concordance <- map_dfr(model_pairs, function(mp) {
  m1 <- mp[1]
  m2 <- mp[2]

  df_compare <- df_values %>%
    filter(model %in% c(m1, m2)) %>%
    select(model, value_name, mean_score, mean_rank, mean_theta) %>%
    pivot_wider(names_from = model, values_from = c(mean_score, mean_rank, mean_theta))

  # Check concordance: if m1 ranks value higher (lower rank), does m1 also score it higher?
  df_compare %<>%
    mutate(
      m1_ranks_higher = .data[[paste0("mean_rank_", m1)]] < .data[[paste0("mean_rank_", m2)]],
      m1_scores_higher = .data[[paste0("mean_score_", m1)]] > .data[[paste0("mean_score_", m2)]],
      concordant = m1_ranks_higher == m1_scores_higher
    )

  tibble(
    model1 = m1,
    model2 = m2,
    n_values = nrow(df_compare),
    concordance_rate = mean(df_compare$concordant, na.rm = TRUE)
  )
})

print(cross_model_concordance)
cat("\nMean cross-model concordance:", round(mean(cross_model_concordance$concordance_rate), 3), "\n")

################################################################################
# 4. RANK CORRELATIONS (Spearman and Kendall)
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 3: Rank Correlations (Spearman/Kendall)\n")
cat(strrep("=", 60), "\n")

# For each model, compute correlation between rank and score
correlations_by_model <- df_long %>%
  group_by(model) %>%
  summarise(
    spearman_rank = cor(rank, score, method = "spearman", use = "complete.obs"),
    kendall_rank = cor(rank, score, method = "kendall", use = "complete.obs"),
    spearman_theta = cor(theta, score, method = "spearman", use = "complete.obs"),
    kendall_theta = cor(theta, score, method = "kendall", use = "complete.obs"),
    pearson_theta = cor(theta, score, method = "pearson", use = "complete.obs"),
    n = n(),
    .groups = "drop"
  )

print(correlations_by_model)

# Note: negative correlation with rank = good (lower rank = more valued = higher score expected)
# Positive correlation with theta = good (higher theta = more valued = higher score expected)

################################################################################
# 5. SCORE DIFFERENCE ~ RANK/THETA DIFFERENCE REGRESSION
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 4: Score Diff ~ Rank/Theta Diff Regressions\n")
cat(strrep("=", 60), "\n")

# Basic OLS
reg_rank_diff <- lm(score_diff ~ rank_diff, data = df_pairs)
reg_theta_diff <- lm(score_diff ~ theta_diff, data = df_pairs)

# With model fixed effects
reg_rank_diff_fe <- feols(score_diff ~ rank_diff | model, data = df_pairs)
reg_theta_diff_fe <- feols(score_diff ~ theta_diff | model, data = df_pairs)

# By model interactions
reg_rank_interact <- feols(score_diff ~ rank_diff * model, data = df_pairs)
reg_theta_interact <- feols(score_diff ~ theta_diff * model, data = df_pairs)

cat("\n--- OLS: score_diff ~ rank_diff ---\n")
summary(reg_rank_diff)

cat("\n--- OLS: score_diff ~ theta_diff ---\n")
summary(reg_theta_diff)

cat("\n--- FE: score_diff ~ rank_diff | model ---\n")
summary(reg_rank_diff_fe)

cat("\n--- FE: score_diff ~ theta_diff | model ---\n")
summary(reg_theta_diff_fe)

# Model summary table
modelsummary(
  list("Rank Diff" = reg_rank_diff,
       "Theta Diff" = reg_theta_diff,
       "Rank + Model FE" = reg_rank_diff_fe,
       "Theta + Model FE" = reg_theta_diff_fe),
  output = "markdown"
)

################################################################################
# 6. BINARY CHOICE PREDICTION (Logistic Regression)
# Pr(v1 wins score) ~ 1(v1 ranked higher) or ~ theta_diff
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 5: Binary Choice Prediction (Logistic)\n")
cat(strrep("=", 60), "\n")

# Logistic regression
logit_rank <- glm(v1_wins_score ~ v1_wins_rank, data = df_pairs, family = binomial)
logit_theta <- glm(v1_wins_score ~ theta_diff, data = df_pairs, family = binomial)
logit_rank_diff <- glm(v1_wins_score ~ rank_diff, data = df_pairs, family = binomial)

cat("\n--- Logit: Pr(v1 wins) ~ 1(v1 ranked higher) ---\n")
summary(logit_rank)

cat("\n--- Logit: Pr(v1 wins) ~ theta_diff ---\n")
summary(logit_theta)

# Pseudo R-squared
null_deviance <- logit_rank$null.deviance
cat("\nMcFadden R² (rank binary):", round(1 - logit_rank$deviance/null_deviance, 4), "\n")
cat("McFadden R² (theta continuous):", round(1 - logit_theta$deviance/null_deviance, 4), "\n")

################################################################################
# 7. CONCORDANCE INDEX (C-STATISTIC / AUC)
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 6: Concordance Index / C-Statistic\n")
cat(strrep("=", 60), "\n")

# Simple C-statistic calculation
# For each pair where outcomes differ, does predictor correctly order them?

calc_c_index <- function(outcome, predictor) {
  n <- length(outcome)
  concordant <- 0
  discordant <- 0
  tied <- 0

  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      if (outcome[i] != outcome[j]) {
        # Outcomes differ, check if predictor orders correctly
        if ((outcome[i] > outcome[j] & predictor[i] > predictor[j]) |
            (outcome[i] < outcome[j] & predictor[i] < predictor[j])) {
          concordant <- concordant + 1
        } else if (predictor[i] == predictor[j]) {
          tied <- tied + 1
        } else {
          discordant <- discordant + 1
        }
      }
    }
  }

  c_index <- (concordant + 0.5 * tied) / (concordant + discordant + tied)
  return(c_index)
}

# Calculate by model (use theta as continuous predictor for score)
c_indices <- df_long %>%
  group_by(model) %>%
  summarise(
    c_index_theta = calc_c_index(score, theta),
    c_index_neg_rank = calc_c_index(score, -rank),  # negative rank so higher = better
    .groups = "drop"
  )

print(c_indices)

################################################################################
# 8. RANK QUARTILE/DECILE ANALYSIS
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 7: Score by Rank Quartile\n")
cat(strrep("=", 60), "\n")

df_long %<>%
  group_by(model) %>%
  mutate(
    rank_quartile = ntile(-rank, 4),  # 4 = highest ranked (lowest rank number)
    rank_decile = ntile(-rank, 10)
  ) %>%
  ungroup()

quartile_analysis <- df_long %>%
  group_by(model, rank_quartile) %>%
  summarise(
    mean_score = mean(score, na.rm = TRUE),
    se_score = sd(score, na.rm = TRUE) / sqrt(n()),
    n = n(),
    .groups = "drop"
  )

print(quartile_analysis %>% pivot_wider(names_from = rank_quartile, values_from = c(mean_score, n)))

# Test for monotonic trend
monotonic_test <- df_long %>%
  group_by(model) %>%
  summarise(
    cor_quartile_score = cor(rank_quartile, score, method = "spearman"),
    .groups = "drop"
  )

print(monotonic_test)

################################################################################
# 9. FIXED EFFECTS REGRESSION WITH CONTROLS
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 8: Fixed Effects Regressions\n")
cat(strrep("=", 60), "\n")

# Score ~ rank/theta with various FE structures
fe_rank <- feols(score ~ rank | model, data = df_long)
fe_theta <- feols(score ~ theta | model, data = df_long)
fe_rank_pair <- feols(score ~ rank | model + pair_id, data = df_long)
fe_theta_pair <- feols(score ~ theta | model + pair_id, data = df_long)

cat("\n--- Score ~ rank | model FE ---\n")
summary(fe_rank)

cat("\n--- Score ~ theta | model FE ---\n")
summary(fe_theta)

cat("\n--- Score ~ rank | model + pair FE ---\n")
summary(fe_rank_pair)

cat("\n--- Score ~ theta | model + pair FE ---\n")
summary(fe_theta_pair)

modelsummary(
  list("Rank" = fe_rank, "Theta" = fe_theta,
       "Rank + Pair FE" = fe_rank_pair, "Theta + Pair FE" = fe_theta_pair),
  output = "markdown"
)

################################################################################
# 10. CROSS-MODEL PREDICTION
# Does Model A's ranking predict Model B's scores?
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 9: Cross-Model Prediction\n")
cat(strrep("=", 60), "\n")

# Create wide format with each model's rank and score
df_cross <- df_long %>%
  select(pair_id, value_pos, model, score, rank, theta, value_name) %>%
  pivot_wider(
    names_from = model,
    values_from = c(score, rank, theta)
  )

# For each model's score, regress on all models' ranks
cross_model_results <- tibble()

for (target_model in c("gpt41", "gemini25", "claude35", "grok4")) {
  for (pred_model in c("gpt41", "gemini25", "claude35", "grok4")) {
    score_col <- paste0("score_", target_model)
    rank_col <- paste0("rank_", pred_model)
    theta_col <- paste0("theta_", pred_model)

    cor_rank <- cor(df_cross[[score_col]], df_cross[[rank_col]],
                    method = "spearman", use = "complete.obs")
    cor_theta <- cor(df_cross[[score_col]], df_cross[[theta_col]],
                     method = "spearman", use = "complete.obs")

    cross_model_results <- bind_rows(cross_model_results, tibble(
      target = target_model,
      predictor = pred_model,
      spearman_rank = cor_rank,
      spearman_theta = cor_theta,
      same_model = target_model == pred_model
    ))
  }
}

print(cross_model_results %>% arrange(target, predictor))

# Summary: own-model vs cross-model prediction
cat("\nOwn-model correlation (theta):",
    mean(cross_model_results$spearman_theta[cross_model_results$same_model]), "\n")
cat("Cross-model correlation (theta):",
    mean(cross_model_results$spearman_theta[!cross_model_results$same_model]), "\n")

################################################################################
# 10b. CROSS-MODEL CONCORDANCE (Absolute & Diff-in-Diff)
# Using ranks only, per scenario
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("METRIC 10: Cross-Model Concordance (Absolute & Relative)\n")
cat(strrep("=", 60), "\n")

# We need data at the scenario level with all models' ranks and scores
# df_cross already has this from above, but let's reshape df for clarity

# Create scenario-level data with v1 and v2 info for all models
df_scenario <- df %>%
  mutate(scenario_id = row_number())

# For cross-model analysis, we need to compare model pairs
model_list <- c("gpt41", "gemini25", "claude35", "grok4")
model_pairs_cross <- combn(model_list, 2, simplify = FALSE)

# ABSOLUTE: For each value in each scenario,
# if rank_modelA(v) < rank_modelB(v), does score_modelA(v) > score_modelB(v)?

absolute_results <- list()

for (mp in model_pairs_cross) {
  m1 <- mp[1]
  m2 <- mp[2]

  # Get rank and score columns for both models
  m1_rank_prefix <- switch(m1, gpt41 = "gpt", gemini25 = "gemini", claude35 = "claude", grok4 = "grok")
  m2_rank_prefix <- switch(m2, gpt41 = "gpt", gemini25 = "gemini", claude35 = "claude", grok4 = "grok")

  for (i in 1:nrow(df_scenario)) {
    row <- df_scenario[i, ]

    # Value 1
    rank_m1_v1 <- row[[paste0("value1_", m1_rank_prefix, "_rank")]]
    rank_m2_v1 <- row[[paste0("value2_", m1_rank_prefix, "_rank")]]  # Wait, this is wrong

    # Actually need to be more careful here
    # value1_gpt_rank is the rank of value1 according to GPT's BT model
    # We want: rank of v1 in model1 vs rank of v1 in model2
  }
}

# Let me redo this more carefully
# For each scenario (row in df), we have:
#   - value1_name, value2_name
#   - value1_{model}_rank, value2_{model}_rank for each model
#   - {model}_v1_score, {model}_v2_score for each model

absolute_cross_results <- tibble()
relative_cross_results <- tibble()

for (mp in model_pairs_cross) {
  m1 <- mp[1]
  m2 <- mp[2]

  m1_rank_prefix <- switch(m1, gpt41 = "gpt", gemini25 = "gemini", claude35 = "claude", grok4 = "grok")
  m2_rank_prefix <- switch(m2, gpt41 = "gpt", gemini25 = "gemini", claude35 = "claude", grok4 = "grok")

  for (i in 1:nrow(df_scenario)) {
    row <- df_scenario[i, ]

    # Ranks (lower = more valued)
    rank_m1_v1 <- row[[paste0("value1_", m1_rank_prefix, "_rank")]]
    rank_m1_v2 <- row[[paste0("value2_", m1_rank_prefix, "_rank")]]
    rank_m2_v1 <- row[[paste0("value1_", m2_rank_prefix, "_rank")]]
    rank_m2_v2 <- row[[paste0("value2_", m2_rank_prefix, "_rank")]]

    # Scores (higher = more valued)
    score_m1_v1 <- row[[paste0(m1, "_v1_score")]]
    score_m1_v2 <- row[[paste0(m1, "_v2_score")]]
    score_m2_v1 <- row[[paste0(m2, "_v1_score")]]
    score_m2_v2 <- row[[paste0(m2, "_v2_score")]]

    # --- ABSOLUTE: Per-value cross-model ---
    # For v1: if rank_m1(v1) < rank_m2(v1), does score_m1(v1) > score_m2(v1)?
    # (lower rank = more valued, so m1 values v1 more if rank is lower)

    # Value 1
    m1_values_v1_more_rank <- rank_m1_v1 < rank_m2_v1  # m1 ranks v1 higher
    m1_scores_v1_higher <- score_m1_v1 > score_m2_v1   # m1 scores v1 higher
    concordant_v1 <- m1_values_v1_more_rank == m1_scores_v1_higher
    abs_rank_diff_v1 <- abs(rank_m1_v1 - rank_m2_v1)

    absolute_cross_results <- bind_rows(absolute_cross_results, tibble(
      model1 = m1, model2 = m2, scenario = i, value = "v1",
      value_name = row$value1_name,
      rank_m1 = rank_m1_v1, rank_m2 = rank_m2_v1,
      score_m1 = score_m1_v1, score_m2 = score_m2_v1,
      m1_ranks_higher = m1_values_v1_more_rank,
      m1_scores_higher = m1_scores_v1_higher,
      concordant = as.integer(concordant_v1),
      abs_rank_diff = abs_rank_diff_v1
    ))

    # Value 2
    m1_values_v2_more_rank <- rank_m1_v2 < rank_m2_v2
    m1_scores_v2_higher <- score_m1_v2 > score_m2_v2
    concordant_v2 <- m1_values_v2_more_rank == m1_scores_v2_higher
    abs_rank_diff_v2 <- abs(rank_m1_v2 - rank_m2_v2)

    absolute_cross_results <- bind_rows(absolute_cross_results, tibble(
      model1 = m1, model2 = m2, scenario = i, value = "v2",
      value_name = row$value2_name,
      rank_m1 = rank_m1_v2, rank_m2 = rank_m2_v2,
      score_m1 = score_m1_v2, score_m2 = score_m2_v2,
      m1_ranks_higher = m1_values_v2_more_rank,
      m1_scores_higher = m1_scores_v2_higher,
      concordant = as.integer(concordant_v2),
      abs_rank_diff = abs_rank_diff_v2
    ))

    # --- RELATIVE (Diff-in-Diff) ---
    # m1's relative preference: rank_m1(v2) - rank_m1(v1)  (positive if m1 prefers v1)
    # m2's relative preference: rank_m2(v2) - rank_m2(v1)
    # If m1_rel_pref > m2_rel_pref, m1 favors v1 more than m2 does
    # Check if scores show same pattern

    m1_rel_rank_pref <- rank_m1_v2 - rank_m1_v1  # positive = m1 prefers v1
    m2_rel_rank_pref <- rank_m2_v2 - rank_m2_v1
    m1_favors_v1_more_rank <- m1_rel_rank_pref > m2_rel_rank_pref

    m1_rel_score_pref <- score_m1_v1 - score_m1_v2  # positive = m1 scores v1 higher
    m2_rel_score_pref <- score_m2_v1 - score_m2_v2
    m1_favors_v1_more_score <- m1_rel_score_pref > m2_rel_score_pref

    concordant_relative <- m1_favors_v1_more_rank == m1_favors_v1_more_score

    # Relative rank diff: how much more does m1 favor v1 over v2 compared to m2?
    rel_rank_diff <- abs(m1_rel_rank_pref - m2_rel_rank_pref)

    relative_cross_results <- bind_rows(relative_cross_results, tibble(
      model1 = m1, model2 = m2, scenario = i,
      value1_name = row$value1_name, value2_name = row$value2_name,
      m1_rel_rank_pref = m1_rel_rank_pref,
      m2_rel_rank_pref = m2_rel_rank_pref,
      m1_rel_score_pref = m1_rel_score_pref,
      m2_rel_score_pref = m2_rel_score_pref,
      m1_favors_v1_more_rank = m1_favors_v1_more_rank,
      m1_favors_v1_more_score = m1_favors_v1_more_score,
      concordant = as.integer(concordant_relative),
      rel_rank_diff = rel_rank_diff
    ))
  }
}

# Summarize ABSOLUTE cross-model concordance by model pair
cat("\n--- ABSOLUTE Cross-Model Concordance (per-value) ---\n")
cat("If BT says model1 values X more than model2, does score agree?\n\n")

absolute_summary <- absolute_cross_results %>%
  group_by(model1, model2) %>%
  summarise(
    n = n(),
    concordance = mean(concordant),
    mean_abs_rank_diff = mean(abs_rank_diff),
    .groups = "drop"
  )

print(absolute_summary)

cat("\nOverall absolute concordance:", mean(absolute_cross_results$concordant), "\n")
cat("Random baseline: 0.50\n")

# Binomial test
binom_abs <- binom.test(sum(absolute_cross_results$concordant), nrow(absolute_cross_results), 0.5)
cat("Binomial test p-value:", format(binom_abs$p.value, scientific = TRUE), "\n")

# Summarize RELATIVE cross-model concordance
cat("\n--- RELATIVE (Diff-in-Diff) Cross-Model Concordance ---\n")
cat("If BT says model1 favors v1 over v2 MORE than model2 does, does score agree?\n\n")

relative_summary <- relative_cross_results %>%
  group_by(model1, model2) %>%
  summarise(
    n = n(),
    concordance = mean(concordant),
    mean_rel_rank_diff = mean(rel_rank_diff),
    .groups = "drop"
  )

print(relative_summary)

cat("\nOverall relative concordance:", mean(relative_cross_results$concordant), "\n")
cat("Random baseline: 0.50\n")

# Binomial test
binom_rel <- binom.test(sum(relative_cross_results$concordant), nrow(relative_cross_results), 0.5)
cat("Binomial test p-value:", format(binom_rel$p.value, scientific = TRUE), "\n")

# Save results
write_csv(absolute_cross_results, "cross_model_absolute_concordance.csv")
write_csv(relative_cross_results, "cross_model_relative_concordance.csv")

################################################################################
# 11. VISUALIZATION
################################################################################

# Plot 1: Concordance rates by model (hardcoded from python script)
concordance_hardcoded <- tibble(
  model = c("claude35", "gemini25", "gpt41", "grok4"),
  n_epochs = c(78, 78, 78, 80),
  concordance_rate_rank = c(0.705128, 0.833333, 0.551282, 0.950000),
  concordance_rate_theta = c(0.705128, 0.833333, 0.551282, 0.950000),
  mean_abs_rank_diff = c(695.769231, 550.679487, 278.192308, 700.150000),
  mean_abs_theta_diff = c(9.776157, 2.571720, 1.051850, 6.642732)
)

p1 <- concordance_hardcoded %>%
  ggplot(aes(x = model, y = concordance_rate_rank, fill = model)) +
  geom_col() +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gold") +
  scale_fill_manual(values = model_colors) +
  labs(title = "Within-model value agreement rate",
       subtitle = "If BT says v1 > v2, do we see the same in Petri?",
       x = "Model", y = "Agreement Rate") +
  myTheme +
  ylim(0, 1) +
  theme(legend.position = "none")
p1
ggsave("plots/concordance_by_model.png", p1, width = 8, height = 6)

# Plot 2: Score vs Theta scatter by model
p2 <- df_long %>%
  ggplot(aes(x = theta, y = score, color = model)) +
  geom_point(alpha = 0.5, size = 1.5) +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~model, scales = "free_x") +
  scale_color_manual(values = model_colors) +
  labs(title = "Experiment Score vs BT Theta",
       subtitle = "Higher theta = BT says more valued; Higher score = experiment says more valued",
       x = "BT Theta", y = "Experiment Score") +
  myTheme +
  theme(legend.position = "none")

ggsave("plots/score_vs_theta.png", p2, width = 10, height = 8)

# Plot 3: Score difference vs Rank difference
# Calculate lm stats for each model
lm_stats <- df_pairs %>%
  group_by(model) %>%
  summarise(
    slope = coef(lm(score_diff ~ theta_diff))[2],
    p_value = summary(lm(score_diff ~ theta_diff))$coefficients[2, 4],
    .groups = "drop"
  ) %>%
  mutate(label = sprintf("slope = %.3f, p = %.3g", slope, p_value))

p3 <- df_pairs %>%
  ggplot(aes(x = theta_diff, y = score_diff, color = model)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE) +
  geom_text(data = lm_stats, aes(x = Inf, y = Inf, label = label, color = model),
            hjust = 1.1, vjust = 1.5, show.legend = FALSE) +
  facet_wrap(~model, scales = "free_x") +
  scale_color_manual(values = model_colors) +
  labs(title = "Theta difference in BT model predicts score difference in petri",
       subtitle = "High theta/petri score means more valued. I wouldn't take the p-values
very seriously since there are only 20 observations",
       x = "BT Theta Difference (v1 - v2)", y = "Petri Score Difference (v1 - v2)") +
  myTheme
p3
ggsave("plots/score_diff_vs_rank_diff.png", p3, width = 10, height = 6)

# Plot 4: Score by rank quartile
p4 <- quartile_analysis %>%
  ggplot(aes(x = factor(rank_quartile), y = mean_score, fill = model)) +
  geom_col(position = "dodge") +
  geom_errorbar(aes(ymin = mean_score - 1.96*se_score, ymax = mean_score + 1.96*se_score),
                position = position_dodge(0.9), width = 0.2) +
  scale_fill_manual(values = model_colors) +
  labs(title = "Mean Score by Rank Quartile",
       subtitle = "Quartile 4 = highest BT rank (most valued)",
       x = "Rank Quartile", y = "Mean Experiment Score") +
  myTheme

ggsave("plots/score_by_quartile.png", p4, width = 10, height = 6)

# Plot 5: Cross-model correlation heatmap
p5 <- cross_model_results %>%
  ggplot(aes(x = predictor, y = target, fill = spearman_theta)) +
  geom_tile() +
  geom_text(aes(label = round(spearman_theta, 2)), color = "white", size = 5) +
  scale_fill_gradient2(low = "#E74C3C", mid = "white", high = "#2ECC71", midpoint = 0) +
  labs(title = "Cross-Model Prediction: Theta → Score",
       subtitle = "Spearman correlation between one model's theta and another's score",
       x = "BT Theta From", y = "Experiment Score Of", fill = "Spearman ρ") +
  myTheme +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("plots/cross_model_heatmap.png", p5, width = 8, height = 6)

# Plot 6: Cross-model concordance bar plot
p6 <- relative_summary %>%
  mutate(pair_label = paste0(model1, "-", model2, "\nRel Rank Diff: ", round(mean_rel_rank_diff))) %>%
  ggplot(aes(x = reorder(pair_label, -concordance), y = concordance, fill = concordance)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.2f", concordance)), vjust = -0.5) +
  scale_fill_gradient(low = "gold", high = "dark green") +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0,1,0.2),
                     labels = scales::percent) +
  labs(title = "Cross-Model Agreement Rate Between BT Model and Petri",
       subtitle = "Relative Rank Difference meausre how much the AIs disagree with each other acorrding to BT Model",
       x = "Model Pair", y = "Agreement Rate") +
  myTheme +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
        legend.position = "none")

ggsave("plots/cross_model_concordance.png", p6, width = 12, height = 6)

################################################################################
# 12. SUMMARY TABLE
################################################################################

cat("\n", strrep("=", 60), "\n")
cat("SUMMARY OF ALL METRICS\n")
cat(strrep("=", 60), "\n")

summary_table <- tibble(
  Metric = c(
    "Overall Concordance (Rank)",
    "Overall Concordance (Theta)",
    "Mean Spearman ρ (Theta-Score)",
    "Mean Spearman ρ (Rank-Score)",
    "Score Diff ~ Rank Diff (β)",
    "Score Diff ~ Theta Diff (β)",
    "Logit McFadden R² (Rank)",
    "Logit McFadden R² (Theta)",
    "Mean C-Index (Theta)",
    "Own-Model Cross-Pred (Theta)",
    "Cross-Model Pred (Theta)"
  ),
  Value = c(
    round(overall_concordance$concordance_rate_rank, 3),
    round(overall_concordance$concordance_rate_theta, 3),
    round(mean(correlations_by_model$spearman_theta), 3),
    round(mean(correlations_by_model$spearman_rank), 3),
    round(coef(reg_rank_diff)["rank_diff"], 5),
    round(coef(reg_theta_diff)["theta_diff"], 3),
    round(1 - logit_rank$deviance/null_deviance, 4),
    round(1 - logit_theta$deviance/null_deviance, 4),
    round(mean(c_indices$c_index_theta), 3),
    round(mean(cross_model_results$spearman_theta[cross_model_results$same_model]), 3),
    round(mean(cross_model_results$spearman_theta[!cross_model_results$same_model]), 3)
  ),
  Interpretation = c(
    "Fraction of pairs where rank ordering matches score ordering",
    "Fraction of pairs where theta ordering matches score ordering",
    "Positive = higher theta → higher score (good)",
    "Negative = lower rank → higher score (good)",
    "Negative = larger rank gap → larger score gap in expected direction",
    "Positive = larger theta gap → larger score gap in expected direction",
    "Variance explained in binary choice by rank",
    "Variance explained in binary choice by theta",
    "Probability of correct ordering given theta",
    "How well own model's BT predicts own scores",
    "How well other models' BT predicts scores"
  )
)

print(summary_table, n = Inf)

# Save summary
write_csv(summary_table, "validation_results_summary.csv")

cat("\n\nAnalysis complete! Plots saved to plots/ directory.\n")
