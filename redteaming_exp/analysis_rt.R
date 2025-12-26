# ==============================================================================
# Red Team Analysis: What explains avg_score?
# ==============================================================================
setwd("~/Documents/aisafety_githubs/values_2_misalignment/redteaming_exp")

if (!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if (!require(fixest)) install.packages("fixest"); library(fixest)
if (!require(modelsummary)) install.packages("modelsummary"); library(modelsummary)
if (!require(magrittr)) install.packages("magrittr"); library(magrittr)

library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)

# Theme for plots
myTheme <- theme(
  plot.title = element_text(size = 15),
  panel.background = element_rect(fill = '#F2F2ED'),
  legend.text = element_text(size = 10),
  plot.subtitle = element_text(size = 12),
  axis.title = element_text(size = 12),
  axis.text = element_text(size = 12, colour = 'black'),
  legend.position = "bottom",
  legend.background = element_rect(linetype = 3, size = 0.5, color = 'black', fill = 'grey94'),
  legend.key = element_rect(size = 0.5, linetype = 1, color = 'black')
)

nicepurp <- "#A88DBF"
niceblue <- '#38A5E0'
nicegreen <- '#A3DCC0'
custom_colors <- c("#2ECC71", "#A3E635", "#F4D03F", "#F39C12", "#E74C3C", "#C0392B", "#0072B2", "#CC79A7")

# ==============================================================================
# Load and prepare data
# ==============================================================================
df <- read_csv("run1_redteam_summary_251210_2341.csv",
               col_types = cols(type_agree = col_character()))

# Clean up the data
df %<>% 
  mutate(
    # Convert booleans (Python "True"/"False" strings)
    type_agree = type_agree == "True",
    # Extract the value source model from type (claude_prefer -> claude)
    value_source_model = str_extract(type, "^(claude|gemini|gpt|grok)"),
    # Simplify target model names
    target_model_short = case_when(
      str_detect(target_model, "claude") ~ "Claude 3.5 Sonnet",
      str_detect(target_model, "gemini") ~ "Gemini 2.5 Pro",
      str_detect(target_model, "gpt") ~ "GPT-4.1-mini",
      str_detect(target_model, "grok") ~ "Grok-4",
      TRUE ~ target_model
    ),
    # Factor versions for regression
    target_model_f = factor(target_model_short),
    query_id_f = factor(query_id),
    source_f = factor(source),
    prefer_or_relative_f = factor(prefer_or_relative),
    value_source_model_f = factor(value_source_model)
  )

# Summary stats
cat("\n========== Summary Statistics ==========\n")
cat("N observations:", nrow(df), "\n")
cat("N unique queries:", n_distinct(df$query_id), "\n")
cat("Avg score overall:", mean(df$avg_score), "\n\n")

# Mean score by key variables
cat("Mean score by target model:\n")
df %>% group_by(target_model_short) %>% 
  summarise(mean_score = mean(avg_score), sd = sd(avg_score), n = n()) %>% print()

cat("\nMean score by type_agree:\n")
df %>% group_by(type_agree) %>% 
  summarise(mean_score = mean(avg_score), sd = sd(avg_score), n = n()) %>% print()

cat("\nMean score by prefer_or_relative:\n")
df %>% group_by(prefer_or_relative) %>% 
  summarise(mean_score = mean(avg_score), sd = sd(avg_score), n = n()) %>% print()

cat("\nMean score by source:\n")
df %>% group_by(source) %>% 
  summarise(mean_score = mean(avg_score), sd = sd(avg_score), n = n()) %>% print()

# ==============================================================================
# PART 1: Full sample analysis - What explains avg_score?
# Cluster SEs at query_id level (unit of variation for queries)
# ==============================================================================
cat("\n\n========== PART 1: Full Sample Regressions (Clustered at query_id) ==========\n")

# Model 1: Just target model
m1_full <- feols(avg_score ~ target_model_f, data = df, cluster = ~query_id)

# Model 2: Just type_agree
m2_full <- feols(avg_score ~ type_agree, data = df, cluster = ~query_id)

# Model 3: Just prefer_or_relative
m3_full <- feols(avg_score ~ prefer_or_relative_f, data = df, cluster = ~query_id)

# Model 4: All main effects
m4_full <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f + source_f, 
                 data = df, cluster = ~query_id)

# Model 5: With value source model
m5_full <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f + source_f + value_source_model_f, 
                 data = df, cluster = ~query_id)

cat("\nFull sample regression results:\n")
etable(m1_full, m2_full, m3_full, m4_full, m5_full,
       headers = c("Target Model", "Type Agree", "Prefer/Relative", "All Main", "With Value Source"))

# ==============================================================================
# PART 2: Subsample - Prefer/Relative only (exclude unmodified)
# ==============================================================================
cat("\n\n========== PART 2: Prefer/Relative Only (Exclude Unmodified) ==========\n")

df_prefer_rel <- df %>% filter(prefer_or_relative != "unmodified")
cat("N observations (prefer/relative only):", nrow(df_prefer_rel), "\n")

m1_pr <- feols(avg_score ~ target_model_f, data = df_prefer_rel, cluster = ~query_id)
m2_pr <- feols(avg_score ~ type_agree, data = df_prefer_rel, cluster = ~query_id)
m3_pr <- feols(avg_score ~ prefer_or_relative_f, data = df_prefer_rel, cluster = ~query_id)
m4_pr <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f + source_f, 
               data = df_prefer_rel, cluster = ~query_id)
m5_pr <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f + source_f + value_source_model_f, 
               data = df_prefer_rel, cluster = ~query_id)

cat("\nPrefer/Relative only regression results:\n")
etable(m1_pr, m2_pr, m3_pr, m4_pr, m5_pr,
       headers = c("Target Model", "Type Agree", "Prefer/Relative", "All Main", "With Value Source"))

# ==============================================================================
# PART 3: Subsample - Source == Tim only
# ==============================================================================
cat("\n\n========== PART 3: Source = Tim Only ==========\n")

df_tim <- df %>% filter(source == "Tim")
cat("N observations (Tim only):", nrow(df_tim), "\n")

m1_tim <- feols(avg_score ~ target_model_f, data = df_tim, cluster = ~query_id)
m2_tim <- feols(avg_score ~ type_agree, data = df_tim, cluster = ~query_id)
m3_tim <- feols(avg_score ~ prefer_or_relative_f, data = df_tim, cluster = ~query_id)
m4_tim <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f, 
                data = df_tim, cluster = ~query_id)
m5_tim <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f + value_source_model_f, 
                data = df_tim, cluster = ~query_id)

cat("\nTim source only regression results:\n")
etable(m1_tim, m2_tim, m3_tim, m4_tim, m5_tim,
       headers = c("Target Model", "Type Agree", "Prefer/Relative", "All Main", "With Value Source"))

# ==============================================================================
# PART 4: Query Fixed Effects - How much residual variation explained?
# Cluster at query_id level for consistency
# ==============================================================================
cat("\n\n========== PART 4: Query Fixed Effects Analysis ==========\n")

# Full sample with query FE
m1_fe <- feols(avg_score ~ target_model_f | query_id, data = df, cluster = ~query_id)
m2_fe <- feols(avg_score ~ type_agree | query_id, data = df, cluster = ~query_id)
m3_fe <- feols(avg_score ~ prefer_or_relative_f | query_id, data = df, cluster = ~query_id)
m4_fe <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f + source_f | query_id, 
               data = df, cluster = ~query_id)
m5_fe <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f + source_f + value_source_model_f | query_id, 
               data = df, cluster = ~query_id)

cat("\nQuery FE regression results (full sample):\n")
etable(m1_fe, m2_fe, m3_fe, m4_fe, m5_fe,
       headers = c("Target Model", "Type Agree", "Prefer/Relative", "All Main", "With Value Source"))

# Prefer/relative only with query FE
cat("\n--- Query FE: Prefer/Relative Only ---\n")
m1_fe_pr <- feols(avg_score ~ target_model_f | query_id, data = df_prefer_rel, cluster = ~query_id)
m2_fe_pr <- feols(avg_score ~ type_agree | query_id, data = df_prefer_rel, cluster = ~query_id)
m3_fe_pr <- feols(avg_score ~ prefer_or_relative_f | query_id, data = df_prefer_rel, cluster = ~query_id)
m4_fe_pr <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f + source_f | query_id, 
                  data = df_prefer_rel, cluster = ~query_id)

etable(m1_fe_pr, m2_fe_pr, m3_fe_pr, m4_fe_pr,
       headers = c("Target Model", "Type Agree", "Prefer/Relative", "All Main"))

# ==============================================================================
# PART 5: Variance decomposition - R² comparison
# ==============================================================================
cat("\n\n========== PART 5: Variance Decomposition (R² Comparison) ==========\n")

# No FE - what explains variation?
cat("\n--- R² without Fixed Effects (Full Sample) ---\n")
r2_df <- tibble(
  Model = c("Target Model Only", "Type Agree Only", "Prefer/Rel Only", "All Main Effects", "With Value Source"),
  R2 = c(r2(m1_full, "r2"), r2(m2_full, "r2"), r2(m3_full, "r2"), r2(m4_full, "r2"), r2(m5_full, "r2")),
  Adj_R2 = c(r2(m1_full, "ar2"), r2(m2_full, "ar2"), r2(m3_full, "ar2"), r2(m4_full, "ar2"), r2(m5_full, "ar2"))
)
print(r2_df)

# With FE - residual variation explained
cat("\n--- Within R² with Query Fixed Effects ---\n")
r2_fe_df <- tibble(
  Model = c("Target Model", "Type Agree", "Prefer/Rel", "All Main", "With Value Source"),
  Within_R2 = c(r2(m1_fe, "wr2"), r2(m2_fe, "wr2"), r2(m3_fe, "wr2"), r2(m4_fe, "wr2"), r2(m5_fe, "wr2"))
)
print(r2_fe_df)

# ==============================================================================
# PART 6: Interaction effects
# ==============================================================================
cat("\n\n========== PART 6: Interaction Effects ==========\n")

# Does type_agree effect vary by target model?
m_interact1 <- feols(avg_score ~ target_model_f * type_agree, data = df, cluster = ~query_id)

# Does type_agree effect vary by prefer vs relative?
m_interact2 <- feols(avg_score ~ prefer_or_relative_f * type_agree, data = df_prefer_rel, cluster = ~query_id)

# Full interaction model
m_interact3 <- feols(avg_score ~ target_model_f * type_agree + prefer_or_relative_f + source_f, 
                     data = df, cluster = ~query_id)

cat("\nInteraction models:\n")
etable(m_interact1, m_interact2, m_interact3,
       headers = c("Model x TypeAgree", "PrefRel x TypeAgree", "Full Interaction"))

# ==============================================================================
# PART 7: Two-way clustering robustness
# ==============================================================================
cat("\n\n========== PART 7: Two-way Clustering Robustness ==========\n")

# Two-way cluster: query_id and target_model
m_twoway <- feols(avg_score ~ target_model_f + type_agree + prefer_or_relative_f + source_f, 
                  data = df, cluster = ~query_id + target_model_f)

cat("\nComparison: One-way vs Two-way clustering:\n")
etable(m4_full, m_twoway, 
       headers = c("Cluster: Query", "Cluster: Query + Model"))

# ==============================================================================
# PART 8: Summary table output
# ==============================================================================
cat("\n\n========== PART 8: Final Summary Table ==========\n")

# Create comprehensive comparison table
final_models <- list(
  "No FE" = m4_full,
  "No FE (Prefer/Rel)" = m4_pr,
  "No FE (Tim)" = m4_tim,
  "Query FE" = m4_fe,
  "Query FE (Prefer/Rel)" = m4_fe_pr
)

etable(final_models, 
       title = "What Explains Red Team Scores?",
       notes = "All SEs clustered at query_id level. Prefer/Rel excludes unmodified rows.")

# ==============================================================================
# PART 9: Visualization
# ==============================================================================
cat("\n\n========== Creating Visualizations ==========\n")
model_colors <- c(
  "GPT-4.1-mini" = "#10A37F",      # OpenAI green
  "Gemini 2.5 Pro" = "#8E75B2",   # Google Gemini purple
  "Claude 3.5 Sonnet" = "#CC9B7A",   # Anthropic beige/tan
  "Grok-4" = "#000000"       # xAI black
)
# Plot 1: Mean score by target model
p1 <- df %>% 
  group_by(target_model_short) %>% 
  summarise(mean_score = mean(avg_score), se = sd(avg_score)/sqrt(n())) %>%
  ggplot(aes(x = reorder(target_model_short, mean_score), y = mean_score, fill = target_model_short)) +
  geom_col() +
  geom_errorbar(aes(
    ymin = mean_score - 1.96*se, ymax = mean_score + 1.96*se), 
    width = 0.2, color = "grey30") +
  scale_y_continuous(limits = c(0,5))+
  coord_flip() +
  scale_fill_manual(values = model_colors) +
  labs(title = "Different LLMs Have Different Refusal Rates",
    subtitle = "Mean safety score by LLM on the full set of questions (handwritten and Claude generated)",
    x = "", y = "Average Score (0-5) 5 = Refusal, 0 = Compliance",
       caption = "The 95% Confidence Interval Might Do Not Account for Clusters") +
  myTheme +
  theme(legend.position = "none")
p1

# Plot 2: Mean score by type_agree
p2 <- df %>% 
  filter(prefer_or_relative != "unmodified") %>%
  group_by(type_agree) %>% 
  summarise(mean_score = mean(avg_score), se = sd(avg_score)/sqrt(n())) %>%
  ggplot(aes(x = factor(type_agree), y = mean_score, fill = factor(type_agree))) +
  geom_col(width = 0.7) +
  scale_y_continuous(limits = c(0,5))+
  scale_fill_manual(values = c( custom_colors[5],custom_colors[1])) +
  geom_errorbar(aes(ymin = mean_score - 1.96*se, ymax = mean_score + 1.96*se), width = 0.2) +
  labs(subtitle = "Mean Score by whether the question is rephrased to align with an LLM's own values", 
       title = "Modifying Queries to be Value-Laden Does Not Change Refusal Rates",
       x = "Question Rephrased to Contain an LLM's Preferred Value?", y = "Average Score (0-5) 5 = Refusal, 0 = Compliance",) +
  myTheme +
  theme(legend.position = "none")
p2


# Plot 3: Mean score by prefer vs relative
p3 <- df %>% 
  group_by(prefer_or_relative) %>% 
  summarise(mean_score = mean(avg_score), se = sd(avg_score)/sqrt(n())) %>%
  ggplot(aes(x = prefer_or_relative, y = mean_score, fill = prefer_or_relative)) +
  geom_col() +
  scale_y_continuous(limits = c(0,5))+
  geom_errorbar(aes(ymin = mean_score - 1.96*se, ymax = mean_score + 1.96*se), width = 0.2) +
  labs(title = "Mean Score by Value Type", x = "", y = "Average Score (0-5) 5 = Refusal, 0 = Compliance",
       subtitle = "The standard errors might not be fully correct here.") +
  myTheme +
  theme(legend.position = "none")

# Plot 4: Heatmap of model x type_agree
p4 <- df %>% 
  filter(prefer_or_relative != "unmodified") %>%
  group_by(target_model_short, type_agree) %>% 
  summarise(mean_score = mean(avg_score), .groups = "drop") %>%
  ggplot(aes(x = target_model_short, y = factor(type_agree), fill = mean_score)) +
  geom_tile() +
  geom_text(aes(label = round(mean_score, 2)), color = "white", size = 5) +
  scale_fill_gradient(low = niceblue, high = nicepurp) +
  labs(title = "Mean Score: Target Model x Type Agreement", 
       x = "Target Model", y = "Type Agree", fill = "Avg Score") +
  myTheme

# Save plots
ggsave("../plots/rt_score_by_model.png", p1, width = 8, height = 5)
ggsave("../plots/rt_score_by_typeagree.png", p2, width = 8, height = 5)
ggsave("plots/rt_score_by_prefrel.png", p3, width = 6, height = 5)
ggsave("plots/rt_score_heatmap.png", p4, width = 8, height = 5)

cat("\nPlots saved to plots/ folder\n")

# ==============================================================================
# PART 10: Within-Model Analysis - Does type_agree/prefer_rel predict WITHIN each model?
# ==============================================================================
cat("\n\n========== PART 10: Within-Model Variation Analysis ==========\n")

# Filter to prefer/relative only for cleaner analysis
df_pr <- df %>% filter(prefer_or_relative != "unmodified")

# --- 10A: Split by model (separate regressions) ---
cat("\n--- 10A: Separate Regressions by Target Model (No Query FE) ---\n")

models_list <- c("claude", "gemini", "gpt", "grok")

# Without query FE
split_nofe <- lapply(models_list, function(m) {
  feols(avg_score ~ type_agree + prefer_or_relative_f + source_f, 
        data = df_pr %>% filter(target_model_short == m), 
        cluster = ~query_id)
})
names(split_nofe) <- models_list

etable(split_nofe, headers = models_list,
       title = "Within-Model: Type Agree & Prefer/Rel Effects (No Query FE)")

# With query FE
cat("\n--- 10B: Separate Regressions by Target Model (With Query FE) ---\n")

split_fe <- lapply(models_list, function(m) {
  feols(avg_score ~ type_agree + prefer_or_relative_f + source_f | query_id, 
        data = df_pr %>% filter(target_model_short == m), 
        cluster = ~query_id)
})
names(split_fe) <- models_list

etable(split_fe, headers = models_list,
       title = "Within-Model: Type Agree & Prefer/Rel Effects (With Query FE)")


# ==============================================================================
# PART 10B-EXTRA: Per-Model Robustness Checks (FULL SAMPLE - including unmodified)
# ==============================================================================
cat("\n\n========== PART 10B-EXTRA: Per-Model Robustness (Full Sample, Various Splits) ==========\n")

models_list <- c("claude", "gemini", "gpt", "grok")

# --- FULL SAMPLE (no filter on prefer/relative) ---
cat("\n--- Full Sample (All rows including unmodified) - No Query FE ---\n")
split_full_nofe <- lapply(models_list, function(m) {
  feols(avg_score ~ type_agree + prefer_or_relative_f + source_f, 
        data = df %>% filter(target_model_short == m), 
        cluster = ~query_id)
})
names(split_full_nofe) <- models_list
etable(split_full_nofe, headers = models_list)

cat("\n--- Full Sample (All rows including unmodified) - With Query FE ---\n")
split_full_fe <- lapply(models_list, function(m) {
  feols(avg_score ~ type_agree + prefer_or_relative_f + source_f | query_id, 
        data = df %>% filter(target_model_short == m), 
        cluster = ~query_id)
})
names(split_full_fe) <- models_list
etable(split_full_fe, headers = models_list)

# --- TIM ONLY ---
cat("\n--- Tim Only - No Query FE ---\n")
df_tim_full <- df %>% filter(source == "Tim")
split_tim_nofe <- lapply(models_list, function(m) {
  d <- df_tim_full %>% filter(target_model_short == m)
  if(nrow(d) < 10) return(NULL)
  feols(avg_score ~ type_agree + prefer_or_relative_f, data = d, cluster = ~query_id)
})
names(split_tim_nofe) <- models_list
split_tim_nofe <- split_tim_nofe[!sapply(split_tim_nofe, is.null)]
if(length(split_tim_nofe) > 0) etable(split_tim_nofe, headers = names(split_tim_nofe))

cat("\n--- Tim Only - With Query FE ---\n")
split_tim_fe <- lapply(models_list, function(m) {
  d <- df_tim_full %>% filter(target_model_short == m)
  if(nrow(d) < 10) return(NULL)
  tryCatch(
    feols(avg_score ~ type_agree + prefer_or_relative_f | query_id, data = d, cluster = ~query_id),
    error = function(e) NULL
  )
})
names(split_tim_fe) <- models_list
split_tim_fe <- split_tim_fe[!sapply(split_tim_fe, is.null)]
if(length(split_tim_fe) > 0) etable(split_tim_fe, headers = names(split_tim_fe))

# --- CLAUDE SOURCE ONLY ---
cat("\n--- Claude Source Only - No Query FE ---\n")
df_claude_source <- df %>% filter(source == "Claude" | type == 'unmodified')
split_claudesrc_nofe <- lapply(models_list, function(m) {
  d <- df_claude_source %>% filter(target_model_short == m)
  if(nrow(d) < 10) return(NULL)
  feols(avg_score ~ type_agree + prefer_or_relative_f, data = d, cluster = ~query_id)
})
names(split_claudesrc_nofe) <- models_list
split_claudesrc_nofe <- split_claudesrc_nofe[!sapply(split_claudesrc_nofe, is.null)]
if(length(split_claudesrc_nofe) > 0) etable(split_claudesrc_nofe, headers = names(split_claudesrc_nofe))

cat("\n--- Claude Source Only - With Query FE ---\n")
split_claudesrc_fe <- lapply(models_list, function(m) {
  d <- df_claude_source %>% filter(target_model_short == m)
  if(nrow(d) < 10) return(NULL)
  tryCatch(
    feols(avg_score ~ type_agree + prefer_or_relative_f | query_id, data = d, cluster = ~query_id),
    error = function(e) NULL
  )
})
names(split_claudesrc_fe) <- models_list
split_claudesrc_fe <- split_claudesrc_fe[!sapply(split_claudesrc_fe, is.null)]
if(length(split_claudesrc_fe) > 0) etable(split_claudesrc_fe, headers = names(split_claudesrc_fe))

# --- RELATIVE & UNMODIFIED ONLY (exclude prefer) ---
cat("\n--- Relative & Unmodified Only - No Query FE ---\n")
df_rel_unmod <- df %>% filter(prefer_or_relative != "prefer")
split_relunmod_nofe <- lapply(models_list, function(m) {
  d <- df_rel_unmod %>% filter(target_model_short == m)
  if(nrow(d) < 10) return(NULL)
  feols(avg_score ~ type_agree + prefer_or_relative_f + source_f, data = d, cluster = ~query_id)
})
names(split_relunmod_nofe) <- models_list
split_relunmod_nofe <- split_relunmod_nofe[!sapply(split_relunmod_nofe, is.null)]
if(length(split_relunmod_nofe) > 0) etable(split_relunmod_nofe, headers = names(split_relunmod_nofe))

cat("\n--- Relative & Unmodified Only - With Query FE ---\n")
split_relunmod_fe <- lapply(models_list, function(m) {
  d <- df_rel_unmod %>% filter(target_model_short == m)
  if(nrow(d) < 10) return(NULL)
  tryCatch(
    feols(avg_score ~ type_agree + prefer_or_relative_f + source_f | query_id, data = d, cluster = ~query_id),
    error = function(e) NULL
  )
})
names(split_relunmod_fe) <- models_list
split_relunmod_fe <- split_relunmod_fe[!sapply(split_relunmod_fe, is.null)]
if(length(split_relunmod_fe) > 0) etable(split_relunmod_fe, headers = names(split_relunmod_fe))

# --- Summary of type_agree coefficient across splits ---
cat("\n\n--- SUMMARY: type_agree Coefficient by Model and Sample Split ---\n")
cat("(Negative = using model's own values makes it MORE vulnerable)\n\n")

# --- 10C: Pooled with Model FE (within-model variation across all) ---
cat("\n--- 10C: Pooled Regressions with Model FE (Within-Model Variation) ---\n")

# Model FE only - captures within-model variation
m_modelfe_1 <- feols(avg_score ~ type_agree | target_model_f, 
                     data = df_pr, cluster = ~query_id)
m_modelfe_2 <- feols(avg_score ~ prefer_or_relative_f | target_model_f, 
                     data = df_pr, cluster = ~query_id)
m_modelfe_3 <- feols(avg_score ~ type_agree + prefer_or_relative_f | target_model_f, 
                     data = df_pr, cluster = ~query_id)
m_modelfe_4 <- feols(avg_score ~ type_agree + prefer_or_relative_f + source_f | target_model_f, 
                     data = df_pr, cluster = ~query_id)

etable(m_modelfe_1, m_modelfe_2, m_modelfe_3, m_modelfe_4,
       headers = c("Type Agree", "Prefer/Rel", "Both", "All"),
       title = "Pooled with Model FE (Within-Model Variation)")

# --- 10D: Pooled with Model FE + Query FE ---
cat("\n--- 10D: Pooled with Model FE + Query FE (Double FE) ---\n")

m_doublefe_1 <- feols(avg_score ~ type_agree | target_model_f + query_id, 
                      data = df_pr, cluster = ~query_id)
m_doublefe_2 <- feols(avg_score ~ prefer_or_relative_f | target_model_f + query_id, 
                      data = df_pr, cluster = ~query_id)
m_doublefe_3 <- feols(avg_score ~ type_agree + prefer_or_relative_f | target_model_f + query_id, 
                      data = df_pr, cluster = ~query_id)
m_doublefe_4 <- feols(avg_score ~ type_agree + prefer_or_relative_f + source_f | target_model_f + query_id, 
                      data = df_pr, cluster = ~query_id)

etable(m_doublefe_1, m_doublefe_2, m_doublefe_3, m_doublefe_4,
       headers = c("Type Agree", "Prefer/Rel", "Both", "All"),
       title = "Pooled with Model FE + Query FE")

# --- 10E: R² comparison for within-model analysis ---
cat("\n--- 10E: Within-Model R² Summary ---\n")

cat("\nWithin R² with Model FE only:\n")
r2_modelfe <- tibble(
  Model = c("Type Agree", "Prefer/Rel", "Both", "All"),
  Within_R2 = c(r2(m_modelfe_1, "wr2"), r2(m_modelfe_2, "wr2"), 
                r2(m_modelfe_3, "wr2"), r2(m_modelfe_4, "wr2"))
)
print(r2_modelfe)

cat("\nWithin R² with Model FE + Query FE:\n")
r2_doublefe <- tibble(
  Model = c("Type Agree", "Prefer/Rel", "Both", "All"),
  Within_R2 = c(r2(m_doublefe_1, "wr2"), r2(m_doublefe_2, "wr2"), 
                r2(m_doublefe_3, "wr2"), r2(m_doublefe_4, "wr2"))
)
print(r2_doublefe)

# --- 10F: By-model R² summary ---
cat("\n--- 10F: R² by Model (How much does type_agree + prefer_rel explain within each model?) ---\n")

r2_bymodel <- tibble(
  Model = models_list,
  R2_noFE = sapply(split_nofe, function(x) r2(x, "r2")),
  R2_queryFE = sapply(split_fe, function(x) r2(x, "wr2"))
)
print(r2_bymodel)

# ==============================================================================
# Final summary print
# ==============================================================================
cat("\n\n========== KEY FINDINGS ==========\n")
cat("1. Target model R² (no FE):", round(r2(m1_full, "r2"), 3), "\n")
cat("2. Type agree R² (no FE):", round(r2(m2_full, "r2"), 3), "\n")
cat("3. Prefer/Relative R² (no FE):", round(r2(m3_full, "r2"), 3), "\n")
cat("4. All main effects R² (no FE):", round(r2(m4_full, "r2"), 3), "\n")
cat("5. Within R² with query FE (all main):", round(r2(m4_fe, "wr2"), 3), "\n")
cat("\n--- Within-Model Analysis ---\n")
cat("6. Type agree within-model R² (model FE only):", round(r2(m_modelfe_1, "wr2"), 4), "\n")
cat("7. Prefer/Rel within-model R² (model FE only):", round(r2(m_modelfe_2, "wr2"), 4), "\n")
cat("8. Both within-model R² (model FE + query FE):", round(r2(m_doublefe_3, "wr2"), 4), "\n")




