# 12b_table1_v2.R - Tablo 1'i v2 sonuçlarından üretir.
# (12_table1.R eski v1 dosyasını okuyordu; bu dosya v2'yi okur.)

setwd("C:/SHAP_imbalance")
library(dplyr)

res <- readRDS("results/full_sim_v2_results.rds")

metrics <- c("kendall", "spearman", "rbo", "jaccard_k5", "jaccard_k7",
             "jaccard_k10", "recall_k5", "recall_k7", "l1", "l2", "cosine")

strategy_map <- c(
  "none" = "None", "class_weight" = "Class weighting",
  "random_over" = "Random oversampling", "random_under" = "Random undersampling",
  "smote" = "SMOTE", "adasyn" = "ADASYN"
)
res$strategy_label <- strategy_map[as.character(res$strategy)]

overall <- res %>%
  filter(!is.na(kendall)) %>%
  group_by(strategy_label) %>%
  summarise(across(all_of(metrics), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

rank_tab <- overall["strategy_label"]
for (m in metrics) {
  rank_tab[[m]] <- if (m %in% c("l1", "l2")) rank(overall[[m]], ties.method = "first")
                   else rank(-overall[[m]], ties.method = "first")
}
names(rank_tab) <- c("Strategy", "Kendall τ", "Spearman ρ", "RBO", "Jaccard@5",
                     "Jaccard@7", "Jaccard@10", "Recall@5", "Recall@7",
                     "L1", "L2", "Cosine")
rank_tab <- rank_tab[order(rank_tab[["Kendall τ"]]), ]

cat("=== Table 1 (v2) ===\n")
print(as.data.frame(rank_tab), row.names = FALSE)

cat("\n=== Ortalamalar (v2) ===\n")
print(as.data.frame(overall %>% mutate(across(where(is.numeric), ~ round(.x, 3)))), row.names = FALSE)

write.csv(rank_tab, "manuscript/table1_v2_ranks.csv", row.names = FALSE)
cat("\nKaydedildi: manuscript/table1_v2_ranks.csv\n")
