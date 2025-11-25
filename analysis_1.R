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


df <- read_csv("bt_value_scores_with_meta.csv")

gpt_4_1_mini <- df %>% filter(model == 'gpt_4_1_mini', !is.na(harm_related))

harm_prioritized <- gpt_4_1_mini %>% 
  filter(rank < 2000, harm_related, moral_alignment == 'Malevolent') %>%
  arrange(rank)
non_harm_deprioritized <- gpt_4_1_mini %>% filter(rank > 2000, !harm_related)



claude_opus_4 <- df %>% filter(model == 'claude_opus_4', !is.na(harm_related))

claude_opus_3 <- df %>% filter(model == 'claude_opus_3', !is.na(harm_related))



harm_prioritized <- claude_opus_4 %>% 
  filter(rank < 2000, harm_related, moral_alignment == 'Malevolent') %>%
  arrange(rank)
non_harm_deprioritized <- claude_opus_4 %>% filter(rank > 2000, !harm_related)


claude_opus_4 %>% write_csv('example_ranks_opus_4.csv')
