# ==============================================================================
# Pairwise Model Comparison Analysis
# For each query_id x type, compare model pairs on safety score
# ==============================================================================
setwd("~/Documents/aisafety_githubs/values_2_misalignment")

library(tidyverse)
library(ggplot2)

# Theme
myTheme <- theme(
  plot.title = element_text(size = 14),
  plot.subtitle = element_text(size = 11),
  panel.background = element_rect(fill = '#F2F2ED'),
  axis.title = element_text(size = 12),
  axis.text = element_text(size = 11, colour = 'black'),
  legend.position = "right"
)

# ==============================================================================
# Load data
# ==============================================================================
df <- read_csv("processed_data/run1_redteam_summary_251210_2341.csv",
               col_types = cols(type_agree = col_character()), show_col_types = FALSE)

df <- df %>% mutate(
  type_agree = type_agree == "True",
  target_model_short = case_when(
    str_detect(target_model, "claude") ~ "claude",
    str_detect(target_model, "gemini") ~ "gemini",
    str_detect(target_model, "gpt") ~ "gpt",
    str_detect(target_model, "grok") ~ "grok"
  )
)

# ==============================================================================
# Function to compute pairwise win rates
# ==============================================================================
compute_pairwise_wins <- function(data, filter_type_agree_for = NULL) {
  # filter_type_agree_for: if not NULL, only include rows where that model has type_agree=TRUE
  
  models <- c("claude", "gemini", "gpt", "grok")
  
  # Reshape to wide: one row per query_id x type, columns for each model's score
  wide <- data %>%
    select(query_id, type, target_model_short, avg_score, type_agree) %>%
    pivot_wider(names_from = target_model_short, values_from = c(avg_score, type_agree),
                names_sep = "_")
  
  # Initialize results matrix
  results <- matrix(NA, nrow = 4, ncol = 4, dimnames = list(models, models))
  
  for (i in 1:4) {
    for (j in 1:4) {
      if (i == j) next
      
      m1 <- models[i]  # row model
      m2 <- models[j]  # column model
      
      score1_col <- paste0("avg_score_", m1)
      score2_col <- paste0("avg_score_", m2)
      type_agree1_col <- paste0("type_agree_", m1)
      
      # Filter data
      subset_data <- wide
      
      # If we want only cases where row model has type_agree
      if (!is.null(filter_type_agree_for) && filter_type_agree_for == m1) {
        subset_data <- subset_data %>% filter(!!sym(type_agree1_col) == TRUE)
      }
      
      # Get scores
      scores <- subset_data %>%
        select(s1 = !!sym(score1_col), s2 = !!sym(score2_col)) %>%
        filter(!is.na(s1), !is.na(s2), s1 != s2)  # exclude ties
      
      if (nrow(scores) == 0) {
        results[i, j] <- NA
      } else {
        # Win rate for row model (higher score = safer = win)
        results[i, j] <- mean(scores$s1 > scores$s2)
      }
    }
  }
  
  return(results)
}

# ==============================================================================
# Function to create heatmap
# ==============================================================================
make_heatmap <- function(mat, title, subtitle = "") {
  # Convert matrix to long format
  df_plot <- as.data.frame(mat) %>%
    rownames_to_column("row_model") %>%
    pivot_longer(-row_model, names_to = "col_model", values_to = "win_rate") %>%
    mutate(
      row_model = factor(row_model, levels = c("claude", "gpt", "gemini", "grok")),
      col_model = factor(col_model, levels = c("claude", "gpt", "gemini", "grok")),
      label = ifelse(is.na(win_rate), "", sprintf("%.0f%%", win_rate * 100))
    )
  
  ggplot(df_plot, aes(x = col_model, y = row_model, fill = win_rate)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = label), color = "black", size = 5) +
    scale_fill_gradient2(low = "#E74C3C", mid = "white", high = "#2ECC71", 
                         midpoint = 0.5, na.value = "grey90",
                         limits = c(0, 1), labels = scales::percent) +
    labs(title = title, subtitle = subtitle,
         x = "Column Model (opponent)", y = "Row Model", 
         fill = "Win Rate\n(row > col)") +
    myTheme +
    coord_fixed()
}

# ==============================================================================
# ANALYSIS 1: Full Sample - Overall pairwise win rates
# ==============================================================================
cat("\n========== FULL SAMPLE: Pairwise Win Rates ==========\n")

wins_full <- compute_pairwise_wins(df)
cat("\nOverall pairwise win rates (row beats column):\n")
print(round(wins_full * 100, 1))

p1 <- make_heatmap(wins_full, "Pairwise Win Rates: Full Sample",
                   "% of query×type where row model scored higher (ties excluded)")
ggsave("plots/pairwise_wins_full.png", p1, width = 7, height = 6)

# ==============================================================================
# ANALYSIS 2: Full Sample - Win rates when row model has type_agree
# ==============================================================================
cat("\n========== FULL SAMPLE: Win Rates When Row Model Has Type Agree ==========\n")

models <- c("claude", "gemini", "gpt", "grok")
wins_type_agree_full <- matrix(NA, nrow = 4, ncol = 4, dimnames = list(models, models))

for (m in models) {
  # Filter to only rows where this model has type_agree=TRUE
  df_filtered <- df %>% 
    filter(target_model_short == m & type_agree == TRUE) %>%
    select(query_id, type) %>%
    distinct()
  
  # Get all model scores for these query_id x type combinations
  df_subset <- df %>% 
    inner_join(df_filtered, by = c("query_id", "type"))
  
  if (nrow(df_subset) > 0) {
    wins_m <- compute_pairwise_wins(df_subset)
    wins_type_agree_full[m, ] <- wins_m[m, ]
  }
}

cat("\nWin rates when row model has type_agree=TRUE:\n")
print(round(wins_type_agree_full * 100, 1))

p2 <- make_heatmap(wins_type_agree_full, "Pairwise Win Rates: When Row Model Has Type Agree",
                   "Full sample - % wins when row model's value is used")
ggsave("plots/pairwise_wins_typeagree_full.png", p2, width = 7, height = 6)

# ==============================================================================
# ANALYSIS 3: Tim Only
# ==============================================================================
cat("\n========== TIM ONLY: Pairwise Win Rates ==========\n")

df_tim <- df %>% filter(source == "Tim")

wins_tim <- compute_pairwise_wins(df_tim)
cat("\nTim only - Overall pairwise win rates:\n")
print(round(wins_tim * 100, 1))

p3 <- make_heatmap(wins_tim, "Pairwise Win Rates: Tim Only",
                   "% of query×type where row model scored higher")
ggsave("plots/pairwise_wins_tim.png", p3, width = 7, height = 6)

# Tim only - with type agree
wins_type_agree_tim <- matrix(NA, nrow = 4, ncol = 4, dimnames = list(models, models))

for (m in models) {
  df_filtered <- df_tim %>% 
    filter(target_model_short == m & type_agree == TRUE) %>%
    select(query_id, type) %>%
    distinct()
  
  df_subset <- df_tim %>% 
    inner_join(df_filtered, by = c("query_id", "type"))
  
  if (nrow(df_subset) > 0) {
    wins_m <- compute_pairwise_wins(df_subset)
    wins_type_agree_tim[m, ] <- wins_m[m, ]
  }
}

cat("\nTim only - Win rates when row model has type_agree=TRUE:\n")
print(round(wins_type_agree_tim * 100, 1))

p4 <- make_heatmap(wins_type_agree_tim, "Pairwise Win Rates: Tim Only + Type Agree",
                   "% wins when row model's value is used")
ggsave("plots/pairwise_wins_typeagree_tim.png", p4, width = 7, height = 6)

# ==============================================================================
# ANALYSIS 4: Relative + Unmodified Only
# ==============================================================================
cat("\n========== REL + UNMOD: Pairwise Win Rates ==========\n")

df_relunmod <- df %>% filter(prefer_or_relative != "prefer")

wins_relunmod <- compute_pairwise_wins(df_relunmod)
cat("\nRel+Unmod - Overall pairwise win rates:\n")
print(round(wins_relunmod * 100, 1))

p5 <- make_heatmap(wins_relunmod, "Pairwise Win Rates: Relative + Unmodified",
                   "% of query×type where row model scored higher")
ggsave("plots/pairwise_wins_relunmod.png", p5, width = 7, height = 6)

# Rel+Unmod - with type agree
wins_type_agree_relunmod <- matrix(NA, nrow = 4, ncol = 4, dimnames = list(models, models))

for (m in models) {
  df_filtered <- df_relunmod %>% 
    filter(target_model_short == m & type_agree == TRUE) %>%
    select(query_id, type) %>%
    distinct()
  
  df_subset <- df_relunmod %>% 
    inner_join(df_filtered, by = c("query_id", "type"))
  
  if (nrow(df_subset) > 0) {
    wins_m <- compute_pairwise_wins(df_subset)
    wins_type_agree_relunmod[m, ] <- wins_m[m, ]
  }
}

cat("\nRel+Unmod - Win rates when row model has type_agree=TRUE:\n")
print(round(wins_type_agree_relunmod * 100, 1))

p6 <- make_heatmap(wins_type_agree_relunmod, "Pairwise Win Rates: Rel+Unmod + Type Agree",
                   "% wins when row model's value is used")
ggsave("plots/pairwise_wins_typeagree_relunmod.png", p6, width = 7, height = 6)

# ==============================================================================
# SUMMARY: Compare overall vs type_agree win rates
# ==============================================================================
cat("\n\n========== SUMMARY: Does Type Agree Help? ==========\n")
cat("(Difference = type_agree win rate - overall win rate)\n")
cat("Positive = type_agree helps the model win more often\n\n")

cat("--- Full Sample ---\n")
diff_full <- wins_type_agree_full - wins_full
print(round(diff_full * 100, 1))

cat("\n--- Tim Only ---\n")
diff_tim <- wins_type_agree_tim - wins_tim
print(round(diff_tim * 100, 1))

cat("\n--- Rel + Unmod ---\n")
diff_relunmod <- wins_type_agree_relunmod - wins_relunmod
print(round(diff_relunmod * 100, 1))

# Create difference heatmap
p7 <- make_heatmap(diff_full, "Type Agree Effect on Win Rate: Full Sample",
                   "Difference: (with type_agree) - (overall). Positive = helps") +
  scale_fill_gradient2(low = "#E74C3C", mid = "white", high = "#2ECC71", 
                       midpoint = 0, na.value = "grey90",
                       limits = c(-0.3, 0.3), labels = scales::percent)
ggsave("plots/pairwise_typeagree_diff_full.png", p7, width = 7, height = 6)

cat("\n\nPlots saved to plots/ folder\n")




