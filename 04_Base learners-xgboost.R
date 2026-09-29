
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
for(i in c(4,14,15)){ 
  Heart[[i]] <- factor(Heart[[i]])
}
# 
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
datarecipe_xgboost <- recipe(formula = Landslides ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = mxjq_ndn)
datarecipe_xgboost
# 
model_xgboost <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  mtry = tune(),
  trees = 1000,
  min_n = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  stop_iter = 25
) %>%
  set_args(validation = 0.2,
           event_level = "second")
model_xgboost
# workflow
wk_xgboost <- 
  workflow() %>%
  add_recipe(datarecipe_xgboost) %>%
  add_model(model_xgboost)
wk_xgboost
##############################################################

#########################  
# 
param_xgboost <- model_xgboost %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 6)))
# 
set.seed(42)
tune_xgboost <- wk_xgboost %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_xgboost,
    metrics = metricset_cls2,
    control = control_bayes(save_pred = T, 
                            verbose = T,
                            no_improve = 10,
                            uncertain = 5,
                            event_level = "second",
                            parallel_over = "everything",
                            save_workflow = T)
  )
########################  
# 
eval_tune_xgboost <- tune_xgboost %>%
  collect_metrics()
eval_tune_xgboost

# autoplot(tune_xgboost)
mxjqcls2_tuneplot(
  eval_tune_xgboost, 
  "roc_auc", 
  "Xgboost"
)
# 
hpbest_xgboost <- tune_xgboost %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_xgboost
# 
set.seed(42)
final_xgboost <- wk_xgboost %>%
  finalize_workflow(hpbest_xgboost) %>%
  fit(traindata)
final_xgboost
##################################################################
#
predtrain_xgboost <- mxjqcls2_predeval(
  model = final_xgboost, 
  dataset = traindata, 
  yname = "Landslides", 
  modelname = "Xgboost", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_xgboost$prediction
predtrain_xgboost$predprobplot
predtrain_xgboost$rocplot
predtrain_xgboost$prplot
predtrain_xgboost$caliplot
predtrain_xgboost$cmplot
predtrain_xgboost$metrics
predtrain_xgboost$diycutoff
predtrain_xgboost$ksplot
#
predtest_xgboost <- mxjqcls2_predeval(
  model = final_xgboost, 
  dataset = testdata, 
  yname = "Landslides", 
  modelname = "Xgboost", 
  datasetname = "testdata",
  cutoff = predtrain_xgboost$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_xgboost$prediction
predtest_xgboost$predprobplot
predtest_xgboost$rocplot
predtest_xgboost$prplot
predtest_xgboost$caliplot
predtest_xgboost$cmplot
predtest_xgboost$metrics
predtest_xgboost$diycutoff
predtest_xgboost$ksplot
# 
pROC::roc.test(predtrain_xgboost$proc, predtest_xgboost$proc)
# 
mxjqcls2_addroc(list(predtrain_xgboost, predtest_xgboost))
# 
mxjqcls2_addpr(list(predtrain_xgboost, predtest_xgboost))
#
mxjqcls2_addcali(list(predtrain_xgboost, predtest_xgboost))

evalcv_xgboost <- mxjqcls2_besthpcv(
  wkflow = wk_xgboost,
  tuneresult = tune_xgboost,
  hpbest = hpbest_xgboost,
  yname = "Landslides",
  modelname = "Xgboost",
  positivelevel = yourpositivelevel
)
evalcv_xgboost$cvroc
evalcv_xgboost$cvpr
evalcv_xgboost$evalcv
###################################################################
######################## fastshap
# shap
shap_xgboost <- mxjqcls2_shapdata(final_xgboost, traindatax, "1")
# shap
shapvip_xgboost <- 
  mxjqcls2_shapvip(shap_xgboost, "Xgboost", showx = c(catvars, convars))
shapvip_xgboost$plot
# shap
mxjqcls2_shaplot(
  shap_xgboost, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "Xgboost"
)
# shap
mxjqcls2_shaplotplus(
  shap_xgboost, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "Xgboost",
  bottomxstart = -0.5,
  bottomxstop = 0.5,
  topxrange = 0.25
)
# 
mxjqcls2_sdpd1(shap_xgboost, catvars, traindatay, "Landslides")
mxjqcls2_sdpd2(shap_xgboost, catvars, traindatay, "Landslides")
mxjqcls2_sdpc1(shap_xgboost, convars, traindatay, "Landslides")
mxjqcls2_sdpc2(shap_xgboost, convars, traindatay, "Landslides")
mxjqcls2_sdpc3(shap_xgboost, convars, traindatay, "Landslides")
# 
shapunity_xgboost <- mxjqcls2_shapunity(shap_xgboost, catvars, convars)
shapunity_xgboost$viplot
shapunity_xgboost$shaplot
# 
mxjqcls2_sdp(shap_xgboost, "Slope", traindatay, "Landslides")
mxjqcls2_sdp(shap_xgboost, "Aspect", traindatay, "Landslides")
# 
shap41_xgboost <- shapviz::shapviz(
  shap_xgboost,
  X = traindatax
)
shapviz::sv_force(shap41_xgboost, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shap41_xgboost, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))

######################################################################
#
save(tune_xgboost,
     predtrain_xgboost,
     predtest_xgboost,
     evalcv_xgboost,
     shapvip_xgboost,
     file = ".\\cls2\\evalresult_xgboost.RData")
# 
final_xgboost_heart <- final_xgboost
traindata_heart <- traindata
save(final_xgboost_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_xgboost_heart.RData")
