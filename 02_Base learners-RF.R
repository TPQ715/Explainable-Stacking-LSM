##############################################################
# install.packages("tidymodels")
library(tidymodels)
source("tidyfuncs4cls2.R")
library(mxjqkit)
library(mxjqcls2)

library(doParallel)
registerDoParallel(
  makePSOCKcluster(
    max(1, (parallel::detectCores(logical = F))-1)
  )
)

# file.choose()
Heart <- readr::read_csv("quick_test.csv")
colnames(Heart) 

for(i in c(4,14,15)){ 
  Heart[[i]] <- factor(Heart[[i]])
}

Heart$OBJECTID <- NULL
Heart$LON<- NULL
Heart$LAT<- NULL

Heart <- na.omit(Heart)
# Heart <- Heart %>%
#   drop_na(Thal)

skimr::skim(Heart)    

yourpositivelevel <- "1"
yournegativelevel <- "0"

levels(Heart$Landslides)
table(Heart$Landslides)
Heart$Landslides <- factor(
  Heart$Landslides,
  levels = c(yournegativelevel, yourpositivelevel)
)
levels(Heart$Landslides)
table(Heart$Landslides)
##############################################################

set.seed(42)
datasplit <- initial_split(Heart, prop = 0.75, strata = Landslides)
traindata <- training(datasplit) %>%
  sample_n(nrow(.))
testdata <- testing(datasplit) %>%
  sample_n(nrow(.))

set.seed(42)
folds <- vfold_cv(traindata, v = 5, strata = Landslides)
folds

datarecipe_rf <- recipe(formula = Landslides ~ ., traindata)
datarecipe_rf

model_rf <- rand_forest(
  mode = "classification",
  engine = "randomForest", # ranger
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_args(importance = T)
model_rf
# workflow
wk_rf <- 
  workflow() %>%
  add_recipe(datarecipe_rf) %>%
  add_model(model_rf)
wk_rf

#########################  Bayes
# 
param_rf <- model_rf %>%
  extract_parameter_set_dials() %>%
  update(mtry = mtry(c(2, 10)),
         trees = trees(c(100, 1000)),
         min_n = min_n(c(7, 55)))
#
set.seed(42)
tune_rf <- wk_rf %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
    param_info = param_rf,
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
eval_tune_rf <- tune_rf %>%
  collect_metrics()
eval_tune_rf
# 
# autoplot(tune_rf)
mxjqcls2_tuneplot(
  eval_tune_rf, 
  "roc_auc", 
  "RF"
)
# 
hpbest_rf <- tune_rf %>%
  select_by_one_std_err(metric = "roc_auc", desc(min_n))
hpbest_rf
# 
set.seed(42)
final_rf <- wk_rf %>%
  finalize_workflow(hpbest_rf) %>%
  fit(traindata)
final_rf
##################################################################
# 
predtrain_rf <- mxjqcls2_predeval(
  model = final_rf, 
  dataset = traindata, 
  yname = "Landslides", 
  modelname = "RF", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_rf$prediction
predtrain_rf$predprobplot
predtrain_rf$rocplot
predtrain_rf$prplot
predtrain_rf$caliplot
predtrain_rf$cmplot
predtrain_rf$metrics
predtrain_rf$diycutoff
predtrain_rf$ksplot
# 
predtest_rf <- mxjqcls2_predeval(
  model = final_rf, 
  dataset = testdata, 
  yname = "Landslides", 
  modelname = "RF", 
  datasetname = "testdata",
  cutoff = predtrain_rf$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_rf$prediction
predtest_rf$predprobplot
predtest_rf$rocplot
predtest_rf$prplot
predtest_rf$caliplot
predtest_rf$cmplot
predtest_rf$metrics
predtest_rf$diycutoff
predtest_rf$ksplot

pROC::roc.test(predtrain_rf$proc, predtest_rf$proc)

mxjqcls2_addroc(list(predtrain_rf, predtest_rf))

mxjqcls2_addpr(list(predtrain_rf, predtest_rf))

mxjqcls2_addcali(list(predtrain_rf, predtest_rf))

evalcv_rf <- mxjqcls2_besthpcv(
  wkflow = wk_rf,
  tuneresult = tune_rf,
  hpbest = hpbest_rf,
  yname = "Landslides",
  modelname = "RF",
  positivelevel = yourpositivelevel
)
evalcv_rf$cvroc
evalcv_rf$cvpr
evalcv_rf$evalcv


###
######################## fastshap包
# shap
shap_rf <- mxjqcls2_shapdata(final_rf, traindatax, "1")
# shap
shapvip_rf <- 
  mxjqcls2_shapvip(shap_rf, "RF", showx = c(catvars, convars))
shapvip_rf$plot
# shap
mxjqcls2_shaplot(
  shap_rf, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "RF"
)
# shap
mxjqcls2_shaplotplus(
  shap_rf, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "RF",
  bottomxstart = -0.5,
  bottomxstop = 0.5,
  topxrange = 0.25
)
# 
mxjqcls2_sdpd1(shap_rf, catvars, traindatay, "Landslides")
mxjqcls2_sdpd2(shap_rf, catvars, traindatay, "Landslides")
mxjqcls2_sdpc1(shap_rf, convars, traindatay, "Landslides")
mxjqcls2_sdpc2(shap_rf, convars, traindatay, "Landslides")
mxjqcls2_sdpc3(shap_rf, convars, traindatay, "Landslides")
# 
shapunity_rf <- mxjqcls2_shapunity(shap_rf, catvars, convars)
shapunity_rf$viplot
shapunity_rf$shaplot
# 
mxjqcls2_sdp(shap_rf, "Slope", traindatay, "Landslides")
mxjqcls2_sdp(shap_rf, "Aspect", traindatay, "Landslides")
# 
shap41_rf <- shapviz::shapviz(
  shap_rf,
  X = traindatax
)
shapviz::sv_force(shap41_rf, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shap41_rf, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))


####
save(tune_rf,
     predtrain_rf,
     predtest_rf,
     evalcv_rf,
     shapvip_rf,
     file = ".\\cls2\\evalresult_rf.RData")
# 
final_rf_heart <- final_rf
traindata_heart <- traindata
save(final_rf_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_rf_heart.RData")
