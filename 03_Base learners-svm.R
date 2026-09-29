
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

# file.choose()
Heart <- readr::read_csv("quick_test.csv")
colnames(Heart) 

for(i in c(4,14,15)){ 
  Heart[[i]] <- factor(Heart[[i]])
}

Heart$ID <- NULL
Heart$OBJECTID <- NULL
Heart$LON <- NULL
Heart$LAT <- NULL
# 
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

datarecipe_svm <- recipe(formula = Landslides ~ ., traindata) %>%
  step_dummy(all_nominal_predictors(), naming = mxjq_ndn) %>%
  step_normalize(all_predictors())
datarecipe_svm

model_svm <- svm_rbf(
  mode = "classification",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune()
)
model_svm
# workflow
wk_svm <- 
  workflow() %>%
  add_recipe(datarecipe_svm) %>%
  add_model(model_svm)
wk_svm


######################### Bayes
# 
set.seed(42)
tune_svm <- wk_svm %>%
  tune_bayes(
    resamples = folds,
    initial = 10,
    iter = 50,
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
eval_tune_svm <- tune_svm %>%
  collect_metrics()
eval_tune_svm
# 图示
# autoplot(tune_svm)
mxjqcls2_tuneplot(
  eval_tune_svm, 
  "roc_auc", 
  "SVM"
)
# 
hpbest_svm <- tune_svm %>%
  select_best(metric = "roc_auc")
hpbest_svm

set.seed(42)
final_svm <- wk_svm %>%
  finalize_workflow(hpbest_svm) %>%
  fit(traindata)
final_svm
##################################################################
# 
predtrain_svm <- mxjqcls2_predeval(
  model = final_svm, 
  dataset = traindata, 
  yname = "Landslides", 
  modelname = "SVM", 
  datasetname = "traindata",
  cutoff = "yueden",
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtrain_svm$prediction
predtrain_svm$predprobplot
predtrain_svm$rocplot
predtrain_svm$prplot
predtrain_svm$caliplot
predtrain_svm$cmplot
predtrain_svm$metrics
predtrain_svm$diycutoff
predtrain_svm$ksplot
# 
predtest_svm <- mxjqcls2_predeval(
  model = final_svm, 
  dataset = testdata, 
  yname = "Landslides", 
  modelname = "SVM", 
  datasetname = "testdata",
  cutoff = predtrain_svm$diycutoff,
  positivelevel = yourpositivelevel,
  negativelevel = yournegativelevel
)
predtest_svm$prediction
predtest_svm$predprobplot
predtest_svm$rocplot
predtest_svm$prplot
predtest_svm$caliplot
predtest_svm$cmplot
predtest_svm$metrics
predtest_svm$diycutoff
predtest_svm$ksplot

pROC::roc.test(predtrain_svm$proc, predtest_svm$proc)

mxjqcls2_addroc(list(predtrain_svm, predtest_svm))

mxjqcls2_addpr(list(predtrain_svm, predtest_svm))

mxjqcls2_addcali(list(predtrain_svm, predtest_svm))

evalcv_svm <- mxjqcls2_besthpcv(
  wkflow = wk_svm,
  tuneresult = tune_svm,
  hpbest = hpbest_svm,
  yname = "Landslides",
  modelname = "SVM",
  positivelevel = yourpositivelevel
)
evalcv_svm$cvroc
evalcv_svm$cvpr
evalcv_svm$evalcv
###################################################################
###################################### 
# shap
shap_svm <- mxjqcls2_shapdata(final_svm, traindatax, "1")
# shap
shapvip_svm <- 
  mxjqcls2_shapvip(shap_svm, "SVM", showx = c(catvars, convars))
shapvip_svm$plot
# shap
mxjqcls2_shaplot(
  shap_svm, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "SVM"
)
# shap
mxjqcls2_shaplotplus(
  shap_svm, 
  showx = c(catvars, convars), 
  flx = catvars,
  modelname = "SVM",
  bottomxstart = -0.5,
  bottomxstop = 0.5,
  topxrange = 0.25
)
# 按变量类型汇总shap图
mxjqcls2_sdpd1(shap_svm, catvars, traindatay, "Landslides")
mxjqcls2_sdpd2(shap_svm, catvars, traindatay, "Landslides")
mxjqcls2_sdpc1(shap_svm, convars, traindatay, "Landslides")
mxjqcls2_sdpc2(shap_svm, convars, traindatay, "Landslides")
mxjqcls2_sdpc3(shap_svm, convars, traindatay, "Landslides")
# 分类变量独热编码之后的shap图
shapunity_svm <- mxjqcls2_shapunity(shap_svm, catvars, convars)
shapunity_svm$viplot
shapunity_svm$shaplot
# 单变量shap依赖图
mxjqcls2_sdp(shap_svm, "Slope", traindatay, "Landslides")
mxjqcls2_sdp(shap_svm, "Aspect", traindatay, "Landslides")
# 单样本shap分解图
shap41_svm <- shapviz::shapviz(
  shap_svm,
  X = traindatax
)
shapviz::sv_force(shap41_svm, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))
shapviz::sv_waterfall(shap41_svm, row_id = 1) +  # 
  theme(text = element_text(family = "serif"))
######################################################################

save(tune_svm,
     predtrain_svm,
     predtest_svm,
     evalcv_svm,
     shapvip_svm,
     file = ".\\cls2\\evalresult_svm.RData")
# 
final_svm_heart <- final_svm
traindata_heart <- traindata
save(final_svm_heart,
     traindata_heart,
     file = ".\\cls2shiny\\shiny_svm_heart.RData")
