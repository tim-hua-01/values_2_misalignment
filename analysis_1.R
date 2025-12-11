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


df <- read_csv("processed_data/bt_value_scores_with_meta.csv")

c35sonnet <- df %>% filter(model == 'claude_3_5_sonnet', !is.na(harm_related))
grok4 <- df %>% filter(model == 'grok_4', !is.na(harm_related))

c_v_grok <- c35sonnet %>% select(-`Unnamed: 0`) %>% inner_join(
grok4 %>% select(value_index, value_name,theta, rank) %>% 
  rename(c(grok_name = value_name, grok_theta = theta, grok_rank = rank)))

c_v_grok %<>% mutate(theta_diff = theta - grok_theta, theta_diff_abs = abs(theta - grok_theta),
                    rank_diff = rank - grok_rank, rank_diff_abs = abs(rank - grok_rank)) %>% 
  filter(value_name == grok_name) %>% 
  relocate(value_name, theta, grok_theta, rank, grok_rank, theta_diff, theta_diff_abs, 
           rank_diff,rank_diff_abs) %>% rename (claude_theta = theta, claude_rank = rank)



gpt_4_1_mini <- df %>% filter(model == 'gpt_4_1_mini', !is.na(harm_related))
gemini25pro <- df %>% filter(model == 'gemini_2_5_pro', !is.na(harm_related))

gpt_v_gemini <- gpt_4_1_mini %>% select(-`Unnamed: 0`) %>% inner_join(
gemini25pro %>% select(value_index, value_name,theta, rank) %>% 
  rename(c(gemini_name = value_name, gemini_theta = theta, gemini_rank = rank)))
gpt_v_gemini %<>% mutate(theta_diff = theta - gemini_theta, theta_diff_abs = abs(theta - gemini_theta),
rank_diff = rank - gemini_rank, rank_diff_abs = abs(rank - gemini_rank)) %>% 
filter(value_name == gemini_name) %>% 
relocate(value_name, theta, gemini_theta, rank, gemini_rank, theta_diff, theta_diff_abs, 
rank_diff,rank_diff_abs) %>% rename (gpt_theta = theta, gpt_rank = rank)



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
