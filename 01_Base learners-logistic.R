
##############################################################
# install.packages("tidymodels")
library(tidymodels)
source("tidyfuncs4cls2.R")
library(mxjqkit)
library(mxjqcls2)
# 
library(doParallel)
registerDoParallel(
  makePSOCKcluster(
    max(1, (parallel::detectCores(logical = F))-1)
  )
)
# 
# file.choose()
Heart <- readr::read_csv("quick_test.csv")
colnames(Heart) 
# 
# 
for(i in c(4,14,15)){ 
  Heart[[i]] <- factor(Heart[[i]])
}
# 
Heart$ID <- NULL
Heart$OBJECTID <- NULL
Heart$LON <- NULL
Heart$LAT <- NULL
# 
Heart <- na.omit(Heart)
# Heart <- Heart %>%
#   drop_na(Thal)
# 
skimr::skim(Heart)    
# 
yourpositivelevel <- "1"
yournegativelevel <- "0"
# 
levels(Heart$Landslides)
table(Heart$Landslides)
Heart$Landslides <- factor(
  Heart$Landslides,
  levels = c(yournegativelevel, yourpositivelevel)
)
levels(Heart$Landslides)
table(Heart$Landslides)
##############################################################
# 
set.seed(42)
datasplit <- initial_split(Heart, prop = 0.75, strata = Landslides)
traindata <- training(datasplit) %>%
  sample_n(nrow(.))
testdata <- testing(datasplit) %>%
  sample_n(nrow(.))
# 
set.seed(42)
folds <- vfold_cv(traindata, v = 5, strata = Landslides)
folds
# 
datarecipe_logistic <- recipe(Landslides ~ ., traindata)
datarecipe_logistic
# 
model_logistic <- logistic_reg(
  mode = "classification",
  engine = "glm"
)
model_logistic
# workflow
wk_logistic <- 
  workflow() %>%
  add_recipe(datarecipe_logistic) %>%
  add_model(model_logistic)
wk_logistic
# 
set.seed(42)
final_logistic <- wk_logistic %>%
  fit(traindata)
final_logistic
##################################################################
# 
predtrain_logistic <- mxjqcls2_predeval(
  model = final_logistic, 
  dataset = traindata, 
  yname = "Landslides", 
  modelname = "Logistic", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_logistic$prediction
predtrain_logistic$predprobplot
predtrain_logistic$rocplot
predtrain_logistic$prplot
predtrain_logistic$caliplot
predtrain_logistic$cmplot
predtrain_logistic$metrics
predtrain_logistic$diycutoff
predtrain_logistic$ksplot
# 
predtest_logistic <- mxjqcls2_predeval(
  model = final_logistic, 
  dataset = testdata, 
  yname = "Landslides", 
  modelname = "Logistic", 
  datasetname = "testdata",
  cutoff = predtrain_logistic$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_logistic$prediction
predtest_logistic$predprobplot
predtest_logistic$rocplot
predtest_logistic$prplot
predtest_logistic$caliplot
predtest_logistic$cmplot
predtest_logistic$metrics
predtest_logistic$diycutoff
predtest_logistic$ksplot

pROC::roc.test(predtrain_logistic$proc, predtest_logistic$proc)

mxjqcls2_addroc(list(predtrain_logistic, predtest_logistic))

mxjqcls2_addpr(list(predtrain_logistic, predtest_logistic))

mxjqcls2_addcali(list(predtrain_logistic, predtest_logistic))
##################################################################
#
set.seed(42)
cv_logistic <- 
  wk_logistic %>%
  fit_resamples(
    folds,
    metrics = metricset_cls2,
    control = control_resamples(save_pred = T,
                                verbose = T,
                                event_level = "second",
                                parallel_over = "everything",
                                save_workflow = T)
  )
cv_logistic

evalcv_logistic <- list()

metrictemp <- metric_set(yardstick::roc_auc, yardstick::pr_auc)
evalcv_logistic$evalcv <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  metrictemp(Landslides, .pred_1, event_level = "second") %>%
  group_by(.metric) %>%
  mutate(model = "Logistic",
         mean = mean(.estimate),
         sd = sd(.estimate)/sqrt(length(folds$splits)))
evalcv_logistic$evalcv

evalcv_logistic$cvroc <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  roc_curve(Landslides, .pred_1, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic$evalcv %>% filter(.metric == "roc_auc"), 
            by = "id") %>%
  mutate(idAUC = paste(id, " ROCAUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    labs(color = "") +
    theme_bw() +
    theme(panel.grid = element_blank(),
          legend.position = c(1,0),
          legend.justification = c(1,0),
          legend.background = element_blank(),
          legend.key = element_blank(), 
          text = element_text(family = "serif"))
evalcv_logistic$cvroc
# PR
evalcv_logistic$cvpr <- 
  collect_predictions(cv_logistic) %>%
  group_by(id) %>%
  pr_curve(Landslides, .pred_1, event_level = "second") %>%
  ungroup() %>%
  left_join(evalcv_logistic$evalcv %>% filter(.metric == "pr_auc"), 
            by = "id") %>%
  mutate(idAUC = paste(id, " PRAUC:", round(.estimate, 4)),
         idAUC = forcats::as_factor(idAUC)) %>%
  ggplot(aes(x = recall, y = precision, color = idAUC)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed", intercept = 1, slope = -1) +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1),
                       breaks = seq(0, 1, by = 0.2),
                       labels = c(0, seq(0.2, 0.8, by = 0.2), 1)) +
    labs(color = "") +
    theme_bw() +
    theme(panel.grid = element_blank(),
          legend.position = c(0,0),
          legend.justification = c(0,0),
          legend.background = element_blank(),
          legend.key = element_blank(), 
          text = element_text(family = "serif"))
evalcv_logistic$cvpr


######shap
######################## fastshap
# shap
shap_logistic <- mxjqcls2_shapdata(final_logistic, traindatax, "1")
# shap
shapvip_logistic <- 
  mxjqcls2_shapvip(shap_logistic, "Logistic", showx = c(catvars, convars))
shapvip_logistic$plot
# shap
mxjqcls2_shaplot(
  shap_logistic, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "Logistic"
)
# shap
mxjqcls2_shaplotplus(
  shap_logistic, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "Logistic",
  bottomxstart = -0.5,
  bottomxstop = 0.5,
  topxrange = 0.25
)
# 
mxjqcls2_sdpd1(shap_logistic, catvars, traindatay, "Landslides")
mxjqcls2_sdpd2(shap_logistic, catvars, traindatay, "Landslides")
mxjqcls2_sdpc1(shap_logistic, convars, traindatay, "Landslides")
mxjqcls2_sdpc2(shap_logistic, convars, traindatay, "Landslides")
mxjqcls2_sdpc3(shap_logistic, convars, traindatay, "Landslides")
# 
shapunity_logistic <- mxjqcls2_shapunity(shap_logistic, catvars, convars)
shapunity_logistic$viplot
shapunity_logistic$shaplot
#
mxjqcls2_sdp(shap_logistic, "Slope", traindatay, "Landslides")
mxjqcls2_sdp(shap_logistic, "Aspect", traindatay, "Landslides")
# 
shap41_logistic <- shapviz::shapviz(
  shap_logistic,
  X = traindatax
)
shapviz::sv_force(shap41_logistic, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shap41_logistic, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))


######################################################################
save(cv_logistic,
     predtrain_logistic,
     predtest_logistic,
     evalcv_logistic,
     shapvip_logistic,
     file = ".\\cls2\\evalresult_logistic.RData")
# 
final_logistic_heart <- final_logistic
traindata_heart <- traindata
save(final_logistic_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_logistic_heart.RData")
