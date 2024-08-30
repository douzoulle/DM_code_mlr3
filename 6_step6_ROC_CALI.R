
# 基于机器学习算法绘制ROC与校准曲线的R包-runway
{
  library(runway)
  library(dcurves)
  library(gtsummary)
  library(tidyverse)
  library(mlr3verse)
  library(tidyverse)
  library(data.table)
  library(iml)
  library(DALEX)
  library(DALEXtra)
  library(survival)
  library(mlr3proba)
  library(mlr3learners.catboost)
  set.seed(2023)
  rm(list = ls())
}

# 读取数据
var <- readRDS("final_var.rds")

train <- read.csv('trainset_smote.csv') %>% 
  .[,c(var, "Distant_Metastasis")]
test <- read.csv('testset.csv') %>% 
  .[,c(var, "Distant_Metastasis")]

# 创建任务
train$Distant_Metastasis <- as.factor(train$Distant_Metastasis)
test$Distant_Metastasis <- as.factor(test$Distant_Metastasis)

task_train <- as_task_classif(train, target = "Distant_Metastasis", positive = "1")
task_test <- as_task_classif(test, target = "Distant_Metastasis", positive = "1")

# 数据预处理
pbp_prep <- 
  po("filter", # 去除高度相关的列
     filter = mlr3filters::flt("find_correlation"), filter.cutoff = 0.3) %>>%
  po("scale", scale = F) %>>% # 中心化
  po("removeconstants") %>>% # 去掉零方差变量 
  po("encode")

# 查看训练集任务信息
task_prep <- pbp_prep$clone()$train(task_train)[[1]]
dim(task_train$data())
task_prep$feature_types

# 选择模型
{
  # 随机森林
  rf_glr <- as_learner(pbp_prep %>>% lrn("classif.ranger", predict_type="prob")) 
  rf_glr$id <- "randomForest"
  
  # svm
  svm_glr <- as_learner(pbp_prep %>>% lrn("classif.svm", predict_type="prob")) 
  svm_glr$id <- "SVM"
  
  # xgb
  xgb_glr <- as_learner(pbp_prep %>>% lrn("classif.xgboost", predict_type="prob")) 
  xgb_glr$id <- "XGboost"
  
  # k近邻
  kknn_glr <- as_learner(pbp_prep %>>% lrn("classif.kknn", predict_type="prob")) 
  kknn_glr$id <- "kknn"
  
  # 带正则化的广义线性模型
  glmnet_glr <- as_learner(pbp_prep %>>% lrn("classif.glmnet", predict_type="prob")) 
  glmnet_glr$id <- "glmnet"
  
  # 逻辑回归
  log_glr <-as_learner(pbp_prep %>>% lrn("classif.log_reg", predict_type="prob")) 
  log_glr$id <- "logistic"
  
  # 朴素贝叶斯
  nb_glr <-as_learner(pbp_prep %>>% lrn("classif.naive_bayes", predict_type="prob")) 
  nb_glr$id <- "NaiveBayes"
  
  # lda
  lda_glr <- as_learner(pbp_prep %>>% lrn("classif.lda", predict_type="prob")) 
  lda_glr$id <- "lda"
  
  # qda
  qda_glr <- as_learner(pbp_prep %>>% lrn("classif.qda", predict_type="prob")) 
  qda_glr$id <- "qda"
  
  # 决策树
  tree_glr <- as_learner(pbp_prep %>>% lrn("classif.rpart", predict_type="prob")) 
  tree_glr$id <- "decisionTree"
  
  # gbm
  gbm_glr <- as_learner(pbp_prep %>>% lrn("classif.svm", predict_type="prob")) 
  gbm_glr$id <- "GBM"
  
  # 神经网络
  nnet_glr <- as_learner(pbp_prep %>>% lrn("classif.nnet", predict_type="prob")) 
  nnet_glr$id <- "nnet"
  
  # CatBoost
  cat_glr <- as_learner(pbp_prep %>>% lrn("classif.catboost", predict_type="prob")) 
  cat_glr$id <- "CatBoost"
}

# 重采样方法：10折交叉验证
cv <- rsmp("repeated_cv", folds = 10, repeats = 10) # 内部10次，外部10次
set.seed(2023)

# 建立多个模型
design = benchmark_grid(
  tasks = task_train,
  learners = list(rf_glr,svm_glr,xgb_glr,kknn_glr,glmnet_glr,log_glr,
                  nb_glr,lda_glr,qda_glr,tree_glr,gbm_glr,nnet_glr,cat_glr),
  resamplings = cv) # 十折交叉验证

# 加速
library(future)
plan("multisession", workers = 8)

# 减少屏幕输出
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")

# 开始运行（benchmark分析）
bmr <- benchmark(design, store_models = T); bmr

# 获得每一个模型的评价指标
measures <- msrs(c("classif.auc", "classif.acc", "classif.ce", "classif.precision", "classif.recall",
                   "classif.sensitivity", "classif.specificity", "classif.bbrier"))
tab = bmr$aggregate(measures); tab

# 模型排序
library(mlr3benchmark)
bma = as_benchmark_aggr(bmr, measures = msr("classif.auc"))
bma$rank_data(minimize = FALSE) # 可以看见，随机森林效果最好

##### 训练集 #####

# 以下合并每个机器学习算法的预测概率
{
  # 随机森林
  rf_glr$train(task_train)
  rf_glr.prediction <- 
    rf_glr$predict(task_train) %>%
    as.data.table() %>%
    # 必须先转化为data.table才能再保存为data.frame
    as_tibble() %>%
    # 第一种方法保存了金标准，后续模型就不保存了
    dplyr::select(row_ids, truth, prob.1) %>%
    # 改名字是为了后续合并以方便区分
    rename(RF = prob.1)
  
  # 支持向量机
  svm_glr$train(task_train)
  svm_glr.prediction <- 
    svm_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(SVM=prob.1)
  
  # XGB
  xgb_glr$train(task_train)
  xgb_glr.prediction <- 
    xgb_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(XGB=prob.1)
  
  # KNN
  kknn_glr$train(task_train)
  kknn_glr.prediction <-
    kknn_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(KNN=prob.1)
  
  # 带正则化的广义线性模型
  glmnet_glr$train(task_train)
  glmnet_glr.prediction <- 
    glmnet_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(GLMNET=prob.1)
  
  # 逻辑回归
  log_glr$train(task_train)
  log_glr.prediction <- 
    log_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(LR=prob.1)
  
  # 朴素贝叶斯
  nb_glr$train(task_train)
  nb_glr.prediction <- 
    nb_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(NB=prob.1)
  
  # lda
  lda_glr$train(task_train)
  lda_glr.prediction <- 
    lda_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(LDA=prob.1)
  
  # qda
  qda_glr$train(task_train)
  qda_glr.prediction <- 
    qda_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(QDA=prob.1)
  
  # 决策树
  tree_glr$train(task_train)
  tree_glr.prediction <- 
    tree_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(DT=prob.1)
  
  # gbm
  gbm_glr$train(task_train)
  gbm_glr.prediction <- 
    gbm_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(GBM=prob.1)
  
  # 神经网络
  nnet_glr$train(task_train)
  nnet_glr.prediction <- 
    nnet_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(NNET=prob.1)
  
  # CatBoost
  cat_glr$train(task_train)
  cat_glr.prediction <- 
    cat_glr$predict(task_train)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(CAT=prob.1)
}

# 合并结果(获得长数据)
all.prediction <- 
  rf_glr.prediction %>%
  left_join(svm_glr.prediction) %>%
  left_join(xgb_glr.prediction) %>%
  left_join(kknn_glr.prediction) %>% 
  left_join(glmnet_glr.prediction) %>%
  left_join(log_glr.prediction) %>%
  left_join(nb_glr.prediction) %>% 
  left_join(lda_glr.prediction) %>%
  left_join(qda_glr.prediction) %>%
  left_join(tree_glr.prediction) %>% 
  left_join(gbm_glr.prediction) %>%
  left_join(nnet_glr.prediction) %>% 
  left_join(cat_glr.prediction) %>% 
  pivot_longer(names_to = "model",
               values_to = "pred",
               cols = c("RF","SVM","XGB","KNN","GLMNET","LR","NB","LDA","QDA","DT","GBM","NNET","CAT"))

all.prediction$truth <- ifelse(all.prediction$truth=="1",1,0)

# 绘制多模型截断概率与其他测试指标的关系
threshperf_plot_multi(all.prediction,
                      outcome = 'truth',
                      prediction = 'pred',
                      model = 'model',
                      positive = "1")
fix(threshperf_plot_multi) # 可以添加几个颜色

source("threshperf_plot_multi_fix.R") # 如果颜色不够
threshperf_plot_multi_fix(all.prediction,
                          outcome = 'truth',
                          prediction = 'pred',
                          model = 'model',
                          positive = "1") # 本次需要十三个颜色
ggsave("all_train.pdf",width = 6, height = 8)

# 绘制多模型校准曲线
cal_plot_multi(all.prediction,
               outcome = 'truth',
               prediction = 'pred',
               model = 'model',
               n_bins = 0,
               show_loess = T,
               positive = "1")
fix(cal_plot_multi) # 可以添加几个颜色

source("cal_plot_multi_fix.R") # 如果颜色不够
cal_plot_multi_fix(all.prediction,
                   outcome = 'truth',
                   prediction = 'pred',
                   model = 'model',
                   n_bins = 0,
                   show_loess = T,
                   positive = "1") # 本次需要十三个颜色
ggsave("cal_train.pdf",width = 8, height = 6)

# 绘制多模型ROC曲线
roc_plot_multi(all.prediction,
               outcome = 'truth',
               prediction = 'pred',
               model = 'model',
               ci = T,
               plot_title = 'Multiple model ROC curves w/CI ribbons')
fix(roc_plot_multi) # 可以添加几个颜色

source("roc_plot_multi_fix.R") # 如果颜色不够
roc_plot_multi_fix(all.prediction,
                   outcome = 'truth',
                   prediction = 'pred',
                   model = 'model',
                   ci = F, # 注意，使用我这个自定义的函数必须填F，因为我把ci部分注释掉了
                   plot_title = 'Multiple ML model ROC curves') # 本次需要十三个颜色
ggsave("roc_train.pdf",width = 8, height = 6)

##### 验证集 #####

# 以下合并每个机器学习算法的预测概率
{
  # 随机森林
  rf_glr$train(task_train)
  rf_glr.prediction_test <- 
    rf_glr$predict(task_test) %>%
    as.data.table() %>%
    # 必须先转化为data.table才能再保存为data.frame
    as_tibble() %>%
    # 第一种方法保存了金标准，后续模型就不保存了
    dplyr::select(row_ids, truth, prob.1) %>%
    # 改名字是为了后续合并以方便区分
    rename(RF = prob.1)
  
  # 支持向量机
  svm_glr$train(task_train)
  svm_glr.prediction_test <- 
    svm_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(SVM=prob.1)
  
  # XGB
  xgb_glr$train(task_train)
  xgb_glr.prediction_test <- 
    xgb_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(XGB=prob.1)
  
  # KNN
  kknn_glr$train(task_train)
  kknn_glr.prediction_test <-
    kknn_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(KNN=prob.1)
  
  # 带正则化的广义线性模型
  glmnet_glr$train(task_train)
  glmnet_glr.prediction_test <- 
    glmnet_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(GLMNET=prob.1)
  
  # 逻辑回归
  log_glr$train(task_train)
  log_glr.prediction_test <- 
    log_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(LR=prob.1)
  
  # 朴素贝叶斯
  nb_glr$train(task_train)
  nb_glr.prediction_test <- 
    nb_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(NB=prob.1)
  
  # lda
  lda_glr$train(task_train)
  lda_glr.prediction_test <- 
    lda_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(LDA=prob.1)
  
  # qda
  qda_glr$train(task_train)
  qda_glr.prediction_test <- 
    qda_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(QDA=prob.1)
  
  # 决策树
  tree_glr$train(task_train)
  tree_glr.prediction_test <- 
    tree_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(DT=prob.1)
  
  # gbm
  gbm_glr$train(task_train)
  gbm_glr.prediction_test <- 
    gbm_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(GBM=prob.1)
  
  # 神经网络
  nnet_glr$train(task_train)
  nnet_glr.prediction_test <- 
    nnet_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(NNET=prob.1)
  
  # CatBoost
  cat_glr$train(task_train)
  cat_glr.prediction_test <- 
    cat_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(CAT=prob.1)
}

# 合并结果(获得长数据)
all.prediction_test <- 
  rf_glr.prediction_test %>%
  left_join(svm_glr.prediction_test) %>%
  left_join(xgb_glr.prediction_test) %>%
  left_join(kknn_glr.prediction_test) %>% 
  left_join(glmnet_glr.prediction_test) %>%
  left_join(log_glr.prediction_test) %>%
  left_join(nb_glr.prediction_test) %>% 
  left_join(lda_glr.prediction_test) %>%
  left_join(qda_glr.prediction_test) %>%
  left_join(tree_glr.prediction_test) %>% 
  left_join(gbm_glr.prediction_test) %>%
  left_join(nnet_glr.prediction_test) %>% 
  left_join(cat_glr.prediction_test) %>% 
  pivot_longer(names_to = "model",
               values_to = "pred",
               cols = c("RF","SVM","XGB","KNN","GLMNET","LR","NB","LDA","QDA","DT","GBM","NNET","CAT"))

all.prediction_test$truth <- ifelse(all.prediction_test$truth=="1",1,0)

# 绘制多模型截断概率与其他测试指标的关系
threshperf_plot_multi_fix(all.prediction_test,
                          outcome = 'truth',
                          prediction = 'pred',
                          model = 'model',
                          positive = "1")
ggsave("all_test.pdf",width = 6, height = 8)

# 绘制多模型校准曲线
cal_plot_multi_fix(all.prediction_test,
                   outcome = 'truth',
                   prediction = 'pred',
                   model = 'model',
                   n_bins = 0,
                   show_loess = T,
                   positive = "1")
ggsave("cal_test.pdf",width = 8, height = 6)

# 绘制多模型ROC曲线
roc_plot_multi_fix(all.prediction_test,
                   outcome = 'truth',
                   prediction = 'pred',
                   model = 'model',
                   ci = F, # 注意，使用我这个自定义的函数必须填F，因为我把ci部分注释掉了
                   plot_title = 'Multiple ML model ROC curves')
ggsave("roc_test.pdf",width = 8, height = 6)

##### 外部验证集 #####

# 外部验证集
ex <- read.csv('ex.csv') %>% 
  .[,c("Chemotherapy", "Grade", "Radiation", "Regional_Lymph_Nodes", 
       "Surgery_Type", "Tumor_Primary_Site", "Distant_Metastasis")]
task_test <- as_task_classif(ex,
                             target = "Distant_Metastasis",
                             positive = "1")

# 以下合并每个机器学习算法的预测概率
{
  # 随机森林
  rf_glr$train(task_train)
  rf_glr.prediction_test <- 
    rf_glr$predict(task_test) %>%
    as.data.table() %>%
    # 必须先转化为data.table才能再保存为data.frame
    as_tibble() %>%
    # 第一种方法保存了金标准，后续模型就不保存了
    dplyr::select(row_ids, truth, prob.1) %>%
    # 改名字是为了后续合并以方便区分
    rename(RF = prob.1)
  
  # 支持向量机
  svm_glr$train(task_train)
  svm_glr.prediction_test <- 
    svm_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(SVM=prob.1)
  
  # XGB
  xgb_glr$train(task_train)
  xgb_glr.prediction_test <- 
    xgb_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(XGB=prob.1)
  
  # KNN
  kknn_glr$train(task_train)
  kknn_glr.prediction_test <-
    kknn_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(KNN=prob.1)
  
  # 带正则化的广义线性模型
  glmnet_glr$train(task_train)
  glmnet_glr.prediction_test <- 
    glmnet_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(GLMNET=prob.1)
  
  # 逻辑回归
  log_glr$train(task_train)
  log_glr.prediction_test <- 
    log_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(LR=prob.1)
  
  # 朴素贝叶斯
  nb_glr$train(task_train)
  nb_glr.prediction_test <- 
    nb_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(NB=prob.1)
  
  # lda
  lda_glr$train(task_train)
  lda_glr.prediction_test <- 
    lda_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(LDA=prob.1)
  
  # qda
  qda_glr$train(task_train)
  qda_glr.prediction_test <- 
    qda_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(QDA=prob.1)
  
  # 决策树
  tree_glr$train(task_train)
  tree_glr.prediction_test <- 
    tree_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(DT=prob.1)
  
  # gbm
  gbm_glr$train(task_train)
  gbm_glr.prediction_test <- 
    gbm_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(GBM=prob.1)
  
  # 神经网络
  nnet_glr$train(task_train)
  nnet_glr.prediction_test <- 
    nnet_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(NNET=prob.1)
  
  # CatBoost
  cat_glr$train(task_train)
  cat_glr.prediction_test <- 
    cat_glr$predict(task_test)%>%
    as.data.table()%>%
    as_tibble()%>%
    dplyr::select(row_ids,prob.1)%>%
    rename(CAT=prob.1)
}

# 合并结果(获得长数据)
all.prediction_test <- 
  rf_glr.prediction_test %>%
  left_join(svm_glr.prediction_test) %>%
  left_join(xgb_glr.prediction_test) %>%
  left_join(kknn_glr.prediction_test) %>% 
  left_join(glmnet_glr.prediction_test) %>%
  left_join(log_glr.prediction_test) %>%
  left_join(nb_glr.prediction_test) %>% 
  left_join(lda_glr.prediction_test) %>%
  left_join(qda_glr.prediction_test) %>%
  left_join(tree_glr.prediction_test) %>% 
  left_join(gbm_glr.prediction_test) %>%
  left_join(nnet_glr.prediction_test) %>% 
  left_join(cat_glr.prediction_test) %>% 
  pivot_longer(names_to = "model",
               values_to = "pred",
               cols = c("RF","SVM","XGB","KNN","GLMNET","LR","NB","LDA","QDA","DT","GBM","NNET","CAT"))

all.prediction_test$truth <- ifelse(all.prediction_test$truth=="1",1,0)

# 绘制多模型截断概率与其他测试指标的关系
threshperf_plot_multi_fix(all.prediction_test,
                          outcome = 'truth',
                          prediction = 'pred',
                          model = 'model',
                          positive = "1")
ggsave("all_ex.pdf",width = 6, height = 8)

# 绘制多模型校准曲线
cal_plot_multi_fix(all.prediction_test,
                   outcome = 'truth',
                   prediction = 'pred',
                   model = 'model',
                   n_bins = 0,
                   show_loess = T,
                   positive = "1")
ggsave("cal_ex.pdf",width = 8, height = 6)

# 绘制多模型ROC曲线
roc_plot_multi_fix(all.prediction_test,
                   outcome = 'truth',
                   prediction = 'pred',
                   model = 'model',
                   ci = F, # 注意，使用我这个自定义的函数必须填F，因为我把ci部分注释掉了
                   plot_title = 'Multiple ML model ROC curves')
ggsave("roc_ex.pdf",width = 8, height = 6)
