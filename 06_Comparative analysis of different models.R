
#############################################################
# remotes::install_github("tidymodels/probably")
library(tidymodels)
library(mxjqcls2)
# 
evalfiles <- list.files(".\\cls2\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)
# 
nmodels <- 5
cols4model <- rainbow(nmodels)  # 
#############################################################
# 
predtrainlist <- list(
  predtrain_logistic,predtrain_rf,
  predtrain_xgboost, predtrain_svm, predtrain_stack
)
# 
evaltrain <- 
  bind_rows(lapply(predtrainlist, "[[", "metrics")) %>%
  mutate(model = forcats::as_factor(model))
evaltrain
# 
evaltrain %>%
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  mxjqcls2_paralplot()
# 
evaltrain %>%
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  mxjqcls2_heatmap()
# rocauc
mxjqcls2_barplot(evaltrain, "roc_auc")
#############################
# ROC
mxjqcls2_addroc(predtrainlist, cols4model)
# PR
mxjqcls2_addpr(predtrainlist, cols4model)
# 
mxjqcls2_addcali(predtrainlist, cols4model)
############################
# 
predtrain <- 
  bind_rows(lapply(predtrainlist, "[[", "prediction")) %>%
  mutate(model = forcats::as_factor(model))
predtrain
# 
predtrain2 <- predtrain %>%
  dplyr::select(-.pred_0) %>%
  mutate(id = rep(1:nrow(predtrain_logistic$prediction), nmodels)) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_1) %>%
  dplyr::select(id, .obs, sort(unique(predtrain$model)))
predtrain2
# DCA
traindca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtrain2)[3:ncol(predtrain2)], 
               collapse = " + "))
),
data = predtrain2,
thresholds = seq(0, 1, by = 0.01)
)
plot(traindca_obj, smooth = T, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on traindata") +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,1),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))
#############################################################
# 
predtestlist <- list(
  predtest_logistic,predtest_rf,
  predtest_xgboost, predtest_svm, predtest_stack
)
# 
evaltest <- 
  bind_rows(lapply(predtestlist, "[[", "metrics")) %>%
  mutate(model = forcats::as_factor(model))
evaltest
# 
evaltest %>%
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  mxjqcls2_paralplot()
# 
evaltest %>%
  filter(!(.metric %in% c("detection_prevalence"))) %>%
  mxjqcls2_heatmap()
# 
mxjqcls2_barplot(evaltest, "roc_auc")
#############################
# 
mxjqcls2_addroc(predtestlist, cols4model)
# 
mxjqcls2_addpr(predtestlist, cols4model)
# 
mxjqcls2_addcali(predtestlist, cols4model)
############################
# 
predtest <- 
  bind_rows(lapply(predtestlist, "[[", "prediction")) %>%
  mutate(model = forcats::as_factor(model))
predtest
# 
predtest2 <- predtest %>%
  dplyr::select(-.pred_0) %>%
  mutate(id = rep(1:nrow(predtest_logistic$prediction), nmodels)) %>%
  pivot_wider(id_cols = c(id, .obs), 
              names_from = model, 
              values_from = .pred_1) %>%
  dplyr::select(id, .obs, sort(unique(predtest$model)))
predtest2
# 
testdca_obj <- dcurves::dca(as.formula(
  paste0(".obs ~ ", 
         paste(colnames(predtest2)[3:ncol(predtest2)], 
               collapse = " + "))
),
data = predtest2,
thresholds = seq(0, 1, by = 0.01)
)
plot(testdca_obj, smooth = T, span = 0.5) +
  scale_color_manual(values = c("black", "grey", cols4model)) +
  labs(title = "DCA on testdata") +
  theme(panel.grid = element_blank(),
        legend.position = "inside",
        legend.justification = c(1,1),
        legend.background = element_blank(),
        legend.key = element_blank(), 
        text = element_text(family = "serif"))
#############################################################
# 
evalcv <- bind_rows(
  lapply(list(evalcv_logistic,evalcv_rf,
              evalcv_xgboost, evalcv_svm), 
         "[[", 
         "evalcv")
) %>%
  mutate(
    model = forcats::as_factor(model),
    modelperf = paste0(
      format(model, width = 5, justify = "left"), " ",
      stringr::str_to_upper(.metric), ": ",
      sprintf("%.2f", mean),"±", sprintf("%.2f", sd)
    )
  )
evalcv
# ROC
mxjqcls2_cvpl(evalcv, "roc_auc")
# PR
mxjqcls2_cvpl(evalcv, "pr_auc")
# 
# ROC
mxjqcls2_cveb(evalcv, "roc_auc")
# PR
mxjqcls2_cveb(evalcv, "pr_auc")
######################################################
# vip
vipdata <- bind_rows(
  lapply(list(shapvip_logistic,shapvip_rf, 
              shapvip_xgboost, shapvip_svm,shapvip_stack), 
         "[[", 
         "data")
) %>%
  mutate(model = forcats::as_factor(model)) %>%
  mutate(importance = 1) %>%
  arrange(model, shap.abs.mean) %>%
  dplyr::group_by(model) %>%
  mutate(importance2 = cumsum(importance)) %>%
  ungroup()
vipdata %>%
  ggplot(aes(x = model,
             y = importance2,
             group = feature)) +
  geom_line(aes(color = feature), linewidth = 8, alpha = 0.5) +
  geom_point(aes(fill = feature), pch = 22, size = 17) +
  geom_text(aes(label = feature), size = 5, family = "serif") +
  labs(x = "", y = "") +
  theme_minimal() +
  theme(legend.position = "none",
        panel.grid.minor = element_blank(),
        text = element_text(family = "serif"),
        axis.text.y = element_blank(),
        axis.text.x = element_text(angle = 45, size = 15, hjust = 1))
vipdata %>%
  ggplot(aes(x = model, 
             y = importance2,
             group = feature, 
             color = feature)) +
  geom_point() +
  geom_line(linewidth = 1) +
  scale_y_continuous(breaks = 1:length(levels(vipdata$feature)),
                     labels = levels(vipdata$feature)) +
  labs(x = "", y = "") +
  theme_minimal() +
  theme(legend.position = "none",
        panel.grid.minor = element_blank(),
        text = element_text(family = "serif"),
        axis.text.y = element_text(size = 15),
        axis.text.x = element_text(angle = 45, size = 15, hjust = 1))
