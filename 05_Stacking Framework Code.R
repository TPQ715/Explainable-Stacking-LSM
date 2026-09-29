
##############################################################
# install.packages("tidymodels")
library(tidymodels)
# library(bonsai)
source("tidyfuncs4cls2.R")
library(mxjqkit)
library(mxjqcls2)
# 多核并行
library(doParallel)
registerDoParallel(
  makePSOCKcluster(
    max(1, (parallel::detectCores(logical = F))-1)
  )
)
###################################################################

# file.choose()
Heart <- readr::read_csv("quick_test.csv")
colnames(Heart)
# 
for(i in c(4,14,15)){
  Heart[[i]] <- factor(Heart[[i]])
}
# 
Heart$OBJECTID <- NULL
Heart$LON <- NULL
Heart$LAT <- NULL
# 
Heart <- na.omit(Heart)
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
################################################################
# 
library(stacks)
##############################
# 
load(".\\cls2\\evalresult_logistic.RData")
load(".\\cls2\\evalresult_rf.RData")
load(".\\cls2\\evalresult_svm.RData")
load(".\\cls2\\evalresult_xgboost.RData")
models_stack <- 
  stacks() %>% 
  add_candidates(cv_logistic) %>%
  add_candidates(tune_rf) %>%
  add_candidates(tune_svm) %>%
  add_candidates(tune_xgboost)
models_stack
##############################
# 
set.seed(42)
meta_stack <- blend_predictions(
  models_stack, 
  penalty = 10^seq(-2, -0.5, length = 20),
  control = control_grid(save_pred = T, 
                         verbose = T,
                         event_level = "second",
                         parallel_over = "everything",
                         save_workflow = T)
)
meta_stack
autoplot(meta_stack) +
  theme_bw() +
  theme(text = element_text(family = "serif"))
# 
set.seed(42)
final_stack <- fit_members(meta_stack)
final_stack
######################################################
# 
predtrain_stack <- mxjqcls2_predeval(
  model = final_stack, 
  dataset = traindata, 
  yname = "Landslides", 
  modelname = "Stacking", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_stack$prediction
predtrain_stack$predprobplot
predtrain_stack$rocplot
predtrain_stack$prplot
predtrain_stack$caliplot
predtrain_stack$cmplot
predtrain_stack$metrics
predtrain_stack$diycutoff
predtrain_stack$ksplot
# 
predtest_stack <- mxjqcls2_predeval(
  model = final_stack, 
  dataset = testdata, 
  yname = "Landslides", 
  modelname = "Stacking", 
  datasetname = "testdata",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_stack$prediction
predtest_stack$predprobplot
predtest_stack$rocplot
predtest_stack$prplot
predtest_stack$caliplot
predtest_stack$cmplot
predtest_stack$metrics
predtest_stack$diycutoff
predtest_stack$ksplot
# 
pROC::roc.test(predtrain_stack$proc, predtest_stack$proc)
# 
mxjqcls2_addroc(list(predtrain_stack, predtest_stack))
# 
mxjqcls2_addpr(list(predtrain_stack, predtest_stack))
# 
mxjqcls2_addcali(list(predtrain_stack, predtest_stack))
###################################################################



# file.choose()
newHeart <- readr::read_csv("quick_test_1.csv")
# 
for(i in c(4,14,15)){ 
  newHeart[[i]] <- factor(newHeart[[i]])
}
# 
newHeart$ID <- NULL
newHeart$OBJECTID <- NULL
newHeart$LON <- NULL
newHeart$LAT <- NULL
# 
newHeart <- na.omit(newHeart)
# newHeart <- newHeart %>%
#   drop_na(Thal)
# 
newHeart$Landslides <- factor(
  newHeart$Landslides,
  levels = c(yournegativelevel, yourpositivelevel)
)
# 预测
predresult <- newHeart %>%
  bind_cols(predict(final_stack, new_data = newHeart, type = "prob"))%>%
  mutate(
    .pred_class = factor(
      ifelse(.pred_1 >= predtrain_stack$diycutoff, 
             yourpositivelevel, 
             yournegativelevel),
      levels = c(yournegativelevel, yourpositivelevel)
    )
  )
# readr::write_excel_csv(predresult, "stacking_result.csv")
# 
prednew_stack <- mxjqcls2_predeval(
  model = final_stack, 
  dataset = newHeart, 
  yname = "Landslides", 
  modelname = "Stacking", 
  datasetname = "newHeart",
  cutoff = predtrain_stack$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
prednew_stack$prediction
prednew_stack$predprobplot
prednew_stack$rocplot
prednew_stack$prplot
prednew_stack$caliplot
prednew_stack$cmplot
prednew_stack$metrics
prednew_stack$diycutoff
prednew_stack$ksplot
###################################################################
# 
colnames(traindata)
traindatax <- traindata %>%
  dplyr::select(-Landslides)
colnames(traindatax)
# 
traindatay <- traindata$Landslides
# 
catvars <- mxjq_gcat(traindatax)
convars <- mxjq_gcon(traindatax)
######################## DALEX
explainer_stack <- DALEXtra::explain_tidymodels(
  final_stack, 
  data = traindatax,
  y = ifelse(traindata$Landslides == yourpositivelevel, 1, 0),
  type = "classification",
  label = "Stacking"
)
# 
vip_stack <- mxjqcls2_vip(explainer_stack, showN = 5)
vip_stack$plot
# 
pdp_stack_cons <- mxjqcls2_pdp(explainer_stack, convars)
pdp_stack_cons$plot
# mxjqcls2_pdp(explainer_stack, "Slope")
# mxjqcls2_pdp(explainer_stack, Aspect)
# mxjqcls2_pdp(explainer_stack, "TWI")
###################################### 
predictor_stack <- iml::Predictor$new(
  final_stack, 
  data = traindatax,
  y = traindata$Landslides,
  predict.function = function(model, newdata){
    predict(model, newdata, type = "prob") %>%
      rename_with(~gsub(".pred_", "", .x))
  },
  type = "prob"
)
# 
interact_stack <- iml::Interaction$new(predictor_stack)
plot(interact_stack) +
  theme_minimal()
interact_stack_1vo <- 
  iml::Interaction$new(predictor_stack, feature = "Slope")
plot(interact_stack_1vo) +
  theme_minimal()
interact_stack_1v1 <- iml::FeatureEffect$new(
  predictor_stack, 
  feature = c("Slope", "Aspect"),
  method = "pdp"
)
plot(interact_stack_1v1) +
  scale_fill_viridis_c() +
  labs(fill = "") +
  theme_minimal()
###################################### lime
explainer_stack <- lime::lime(
  traindatax,
  lime::as_classifier(final_stack, c(yournegativelevel, yourpositivelevel))
)
explanation_stack <- lime::explain(
  traindatax[1,],  # 
  explainer_stack, 
  n_labels = 2, 
  n_features = ncol(traindatax)
)
lime::plot_features(explanation_stack)
######################## fastshap
# shap
shap_stack <- mxjqcls2_shapdata(final_stack, traindatax, "1")
# shap性
shapvip_stack <- 
  mxjqcls2_shapvip(shap_stack, "Stacking", showx = c(catvars, convars))
shapvip_stack$plot
# shap
mxjqcls2_shaplot(
  shap_stack, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "Stacking"
)
# shap
mxjqcls2_shaplotplus(
  shap_stack, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "Stacking",
  bottomxstart = -0.5,
  bottomxstop = 0.5,
  topxrange = 0.25
)
# 
mxjqcls2_sdpd1(shap_stack, catvars, traindatay, "Landslides")
mxjqcls2_sdpd2(shap_stack, catvars, traindatay, "Landslides")
mxjqcls2_sdpc1(shap_stack, convars, traindatay, "Landslides")
mxjqcls2_sdpc2(shap_stack, convars, traindatay, "Landslides")
mxjqcls2_sdpc3(shap_stack, convars, traindatay, "Landslides")
# 
shapunity_stack <- mxjqcls2_shapunity(shap_stack, catvars, convars)
shapunity_stack$viplot
shapunity_stack$shaplot
# 
mxjqcls2_sdp(shap_stack, "Aspect", traindatay, "Landslides")
mxjqcls2_sdp(shap_stack, "Slope", traindatay, "Landslides")
# 
shap41_stack <- shapviz::shapviz(
  shap_stack,
  X = traindatax
)
shapviz::sv_force(shap41_stack, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shap41_stack, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))
######################################################################
# 
save(predtrain_stack,
     predtest_stack,
     shapvip_stack,
     file = ".\\cls2\\evalresult_stack.RData")
# 
final_stack_heart <- final_stack
traindata_heart <- traindata
save(final_stack_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_stack_heart.RData")
