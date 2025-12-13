setwd("~/Documents/aisafety_githubs/values_2_misalignment")

if (!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if (!require(magrittr)) install.packages("magrittr"); library(magrittr)

library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(skimr)
library(modelsummary)
library(fixest)
library(ggtext)
library(broom)
library(ks)
library(ggpattern)
library(readr)
myTheme <- theme(plot.title = element_text(size = 15),
                 panel.background = element_rect(fill = '#F2F2ED'),
                 legend.text = element_text(size = 10),
                 plot.subtitle = element_text(size = 12),
                 axis.title = element_text(size = 12),
                 axis.text = element_text(size = 12, colour = 'black'),
                 legend.position = "bottom",
                 legend.background = element_rect(linetype = 3,size = 0.5, color = 'black', fill = 'grey94'),
                 legend.key = element_rect(size = 0.5, linetype = 1, color = 'black'))







#I also have some nice colors that I use in my various graphs.
nicepurp <- "#A88DBF"
niceblue <- '#38A5E0'
nicegreen <- '#A3DCC0'

custom_colors <- c("#2ECC71", "#A3E635", "#F4D03F", "#F39C12", "#E74C3C", "#C0392B", "#0072B2", "#CC79A7")


df <- read_csv("processed_data/bt_value_scores_with_meta.csv")#read_csv("processed_data/bt_value_scores_aggregated_with_meta.csv")

#new version c35sonnet <- df %>% filter(model == 'claude_3_5_sonnet',freq_x > 30)
c35sonnet <- df %>% filter(model == 'claude_3_5_sonnet',!is.na(harm_related)) %>%
  mutate(rank_new = rank(-theta, ties.method = "first"),
         rank_og = rank)
grok4 <- df %>% filter(model == 'grok_4', !is.na(harm_related)) %>%
  mutate(rank_new = rank(-theta, ties.method = "first"),
         rank_og = rank)

c_v_grok <- c35sonnet %>% select(-`Unnamed: 0`, -rank) %>% inner_join(
grok4 %>% select(value_index, value_name, theta, rank_new, rank_og) %>% 
  rename(c(grok_name = value_name, grok_theta = theta, grok_rank = rank_new, grok_rank_og = rank_og)))

c_v_grok %<>% mutate(theta_diff = theta - grok_theta, theta_diff_abs = abs(theta - grok_theta),
                    rank_diff = rank_new - grok_rank, rank_diff_abs = abs(rank_new - grok_rank)) %>% 
  filter(value_name == grok_name) %>% 
  relocate(value_name, theta, grok_theta, rank_new, grok_rank, theta_diff, theta_diff_abs, 
           rank_diff, rank_diff_abs) %>% rename(claude_theta = theta, claude_rank = rank_new, claude_rank_og = rank_og)



gpt_4_1_mini <- df %>% filter(model == 'gpt_4_1_mini', !is.na(harm_related)) %>%
  mutate(rank_new = rank(-theta, ties.method = "first"),
         rank_og = rank)
gemini25pro <- df %>% filter(model == 'gemini_2_5_pro', !is.na(harm_related)) %>%
  mutate(rank_new = rank(-theta, ties.method = "first"),
         rank_og = rank)

gpt_v_gemini <- gpt_4_1_mini %>% select(-`Unnamed: 0`, -rank) %>% inner_join(
gemini25pro %>% select(value_index, value_name, theta, rank_new, rank_og) %>% 
  rename(c(gemini_name = value_name, gemini_theta = theta, gemini_rank = rank_new, gemini_rank_og = rank_og)))
gpt_v_gemini %<>% mutate(theta_diff = theta - gemini_theta, theta_diff_abs = abs(theta - gemini_theta),
rank_diff = rank_new - gemini_rank, rank_diff_abs = abs(rank_new - gemini_rank)) %>% 
filter(value_name == gemini_name) %>% 
relocate(value_name, theta, gemini_theta, rank_new, gemini_rank, theta_diff, theta_diff_abs, 
rank_diff, rank_diff_abs) %>% rename(gpt_theta = theta, gpt_rank = rank_new, gpt_rank_og = rank_og)


#values_from_four_models theta and ranking
four_mods <- gpt_v_gemini %>% left_join(
  c_v_grok %>% select(value_name, claude_theta, claude_rank, claude_rank_og, grok_theta, grok_rank, grok_rank_og)) %>%
  rowwise() %>%
  mutate(theta_sd = sd(c(gpt_theta, gemini_theta, claude_theta, grok_theta)),
         rank_sd = sd(c(gpt_rank, gemini_rank, claude_rank, grok_rank)),
         rank_sd_og = sd(c(gpt_rank_og, gemini_rank_og, claude_rank_og, grok_rank_og))) %>%
  ungroup() %>%
  relocate(value_name, theta_sd, rank_sd, 
           gpt_rank, gemini_rank, claude_rank, grok_rank,
           gpt_rank_og, gemini_rank_og, claude_rank_og, grok_rank_og,
           gpt_theta, gemini_theta, claude_theta, grok_theta) %>% arrange(desc(rank_sd))




# New version using aggregated data with freq_x > 30 filter
df_agg <- read_csv("processed_data/bt_value_scores_aggregated_with_meta.csv")

c35sonnet_agg <- df_agg %>% filter(model == 'claude_3_5_sonnet', freq_x > 30) %>%
  mutate(rank_new = rank(-theta, ties.method = "first"),
         rank_og = rank)
grok4_agg <- df_agg %>% filter(model == 'grok_4', freq_x > 30) %>%
  mutate(rank_new = rank(-theta, ties.method = "first"),
         rank_og = rank)

c_v_grok_agg <- c35sonnet_agg %>% select(-`Unnamed: 0`, -rank) %>% inner_join(
  grok4_agg %>% select(value_index, value_name, theta, rank_new, rank_og) %>% 
    rename(c(grok_name = value_name, grok_theta = theta, grok_rank = rank_new, grok_rank_og = rank_og)))

c_v_grok_agg %<>% mutate(theta_diff = theta - grok_theta, theta_diff_abs = abs(theta - grok_theta),
                         rank_diff = rank_new - grok_rank, rank_diff_abs = abs(rank_new - grok_rank)) %>% 
  filter(value_name == grok_name) %>% 
  relocate(value_name, theta, grok_theta, rank_new, grok_rank, theta_diff, theta_diff_abs, 
           rank_diff, rank_diff_abs) %>% rename(claude_theta = theta, claude_rank = rank_new, claude_rank_og = rank_og)

gpt_4_1_mini_agg <- df_agg %>% filter(model == 'gpt_4_1_mini', freq_x > 30) %>%
  mutate(rank_new = rank(-theta, ties.method = "first"),
         rank_og = rank)
gemini25pro_agg <- df_agg %>% filter(model == 'gemini_2_5_pro', freq_x > 30) %>%
  mutate(rank_new = rank(-theta, ties.method = "first"),
         rank_og = rank)

gpt_v_gemini_agg <- gpt_4_1_mini_agg %>% select(-`Unnamed: 0`, -rank) %>% inner_join(
  gemini25pro_agg %>% select(value_index, value_name, theta, rank_new, rank_og) %>% 
    rename(c(gemini_name = value_name, gemini_theta = theta, gemini_rank = rank_new, gemini_rank_og = rank_og)))

gpt_v_gemini_agg %<>% mutate(theta_diff = theta - gemini_theta, theta_diff_abs = abs(theta - gemini_theta),
                             rank_diff = rank_new - gemini_rank, rank_diff_abs = abs(rank_new - gemini_rank)) %>% 
  filter(value_name == gemini_name) %>% 
  relocate(value_name, theta, gemini_theta, rank_new, gemini_rank, theta_diff, theta_diff_abs, 
           rank_diff, rank_diff_abs) %>% rename(gpt_theta = theta, gpt_rank = rank_new, gpt_rank_og = rank_og)

# New four_mods using aggregated data with freq_x > 30
four_mods_agg <- gpt_v_gemini_agg %>% left_join(
  c_v_grok_agg %>% select(value_name, claude_theta, claude_rank, claude_rank_og, grok_theta, grok_rank, grok_rank_og)) %>%
  rowwise() %>%
  mutate(theta_sd = sd(c(gpt_theta, gemini_theta, claude_theta, grok_theta)),
         rank_sd = sd(c(gpt_rank, gemini_rank, claude_rank, grok_rank)),
         rank_sd_og = sd(c(gpt_rank_og, gemini_rank_og, claude_rank_og, grok_rank_og))) %>%
  ungroup() %>%
  relocate(value_name, theta_sd, rank_sd, 
           gpt_rank, gemini_rank, claude_rank, grok_rank,
           gpt_rank_og, gemini_rank_og, claude_rank_og, grok_rank_og,
           gpt_theta, gemini_theta, claude_theta, grok_theta) %>% arrange(desc(rank_sd))

four_mods %>% slice_head(n = 20) %>% select(
  value_name, gpt_rank, gemini_rank, claude_rank, grok_rank
) %>% write_csv('conflicting_values.csv')


four_mods_agg %>% slice_head(n = 20) %>% select(
  value_name, gpt_rank, gemini_rank, claude_rank, grok_rank
) %>% write_csv('conflicting_values_agg.csv')

#old stuff
merged_values <- read_csv('merged_label_values.csv')

four_mods %<>% inner_join(merged_values)
four_mods %>% count(merged_value_names) %>% ggplot() + geom_histogram(aes(x = n))
four_mods %>% count(merged_value_names_granular) %>% ggplot() + geom_histogram(aes(x = n))






harm_prioritized <- gpt_4_1_mini %>% 
  filter(rank < 2000, harm_related, moral_alignment == 'Malevolent') %>%
  arrange(rank)
non_harm_deprioritized <- gpt_4_1_mini %>% filter(rank > 2000, !harm_related)



View(gemini25pro %>% filter(!harm_related) %>% arrange(desc(rank)))


claude_opus_4 <- df %>% filter(model == 'claude_opus_4', !is.na(harm_related))

claude_opus_3 <- df %>% filter(model == 'claude_opus_3', !is.na(harm_related))



harm_prioritized <- claude_opus_4 %>% 
  filter(rank < 2000, harm_related, moral_alignment == 'Malevolent') %>%
  arrange(rank)
non_harm_deprioritized <- claude_opus_4 %>% filter(rank > 2000, !harm_related)


claude_opus_4 %>% write_csv('example_ranks_opus_4.csv')
