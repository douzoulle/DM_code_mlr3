
library(magrittr)
library(tidyverse)
library(randomForest) # 执行随机森林算法
library(varSelRF) # 挑选变量
library(pROC) # 绘制ROC曲线
library(funModeling) # 数据分析、数据准备和模型表现评价
rm(list = ls())

# 读取数据
var <- readRDS("final_var.rds")

train <- read.csv('trainset_smote.csv') %>% 
  .[,c(var, "Distant_Metastasis")]
test <- read.csv('testset.csv') %>% 
  .[,c(var, "Distant_Metastasis")]

df <- train


##### 多模型：DALEX ######
library("DALEX")
library("DALEXtra")
library(mlr3) # 主体包
library(mlr3viz) # 执行可视化功能
library(mlr3learners) # 提供额外学习器
library(mlr3verse) # 扩展包
library(mlr3tuning) # 调整参数
library(mlr3learners.catboost)

for (i in names(df)[c(1:7)]){df[,i] <- as.numeric(df[,i])}

# 创建任务
df$Distant_Metastasis <- as.factor(df$Distant_Metastasis)
test$Distant_Metastasis <- as.factor(test$Distant_Metastasis)

task_train <- as_task_classif(df, target = "Distant_Metastasis", positive = "1")
task_test <- as_task_classif(test, target = "Distant_Metastasis", positive = "1")

# 数据预处理
pbp_prep <- 
  po("scale", scale = F) %>>% # 中心化
  po("removeconstants") # 去掉零方差变量

# 选择多个模型
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

# 训练模型
{
  rf_glr$train(task_train)
  svm_glr$train(task_train)
  xgb_glr$train(task_train)
  kknn_glr$train(task_train)
  glmnet_glr$train(task_train)
  log_glr$train(task_train)
  nb_glr$train(task_train)
  lda_glr$train(task_train)
  qda_glr$train(task_train)
  tree_glr$train(task_train)
  gbm_glr$train(task_train)
  nnet_glr$train(task_train)
  cat_glr$train(task_train)
}

# 在开始解释模型行为之前，先创建一个解释器
{
  ranger_exp <- explain_mlr3(rf_glr,
                             data = df[-7],
                             y = df$Distant_Metastasis,
                             label = "Ranger RF",
                             colorize = FALSE)
  
  svm_exp <- explain_mlr3(svm_glr,
                          data = df[-7],
                          y = df$Distant_Metastasis,
                          label = "SVM",
                          colorize = FALSE)
  
  xgb_exp <- explain_mlr3(xgb_glr,
                          data = df[-7],
                          y = df$Distant_Metastasis,
                          label = "XGB",
                          colorize = FALSE)
  
  kknn_exp <- explain_mlr3(kknn_glr,
                           data = df[-7],
                           y = df$Distant_Metastasis,
                           label = "KNN",
                           colorize = FALSE)
  
  glmnet_exp <- explain_mlr3(glmnet_glr,
                             data = df[-7],
                             y = df$Distant_Metastasis,
                             label = "glmnet",
                             colorize = FALSE)
  
  log_exp <- explain_mlr3(log_glr,
                          data = df[-7],
                          y = df$Distant_Metastasis,
                          label = "LR",
                          colorize = FALSE)
  
  
  nb_exp <- explain_mlr3(nb_glr,
                         data = df[-7],
                         y = df$Distant_Metastasis,
                         label = "Naive Bayes",
                         colorize = FALSE)
  
  
  lda_exp <- explain_mlr3(lda_glr,
                          data = df[-7],
                          y = df$Distant_Metastasis,
                          label = "LDA",
                          colorize = FALSE)
  
  qda_exp <- explain_mlr3(qda_glr,
                          data = df[-7],
                          y = df$Distant_Metastasis,
                          label = "QDA",
                          colorize = FALSE)
  
  tree_exp <- explain_mlr3(tree_glr,
                           data = df[-7],
                           y = df$Distant_Metastasis,
                           label = "Decision Tree",
                           colorize = FALSE)
  
  gbm_exp <- explain_mlr3(gbm_glr,
                          data = df[-7],
                          y = df$Distant_Metastasis,
                          label = "GBM",
                          colorize = FALSE)
  
  nnet_exp <- explain_mlr3(nnet_glr,
                           data = df[-7],
                           y = df$Distant_Metastasis,
                           label = "NNET",
                           colorize = FALSE)
  
  cat_exp <- explain_mlr3(cat_glr,
                          data = df[-7],
                          y = df$Distant_Metastasis,
                          label = "CAT",
                          colorize = FALSE)
}

# 粗略看看重要性
plot(model_parts(cat_exp))
plot(model_profile(cat_exp, 
                   variable = "Grade", # 可以每个自变量都尝试下
                   type = "partial"))

# 粗略看看shap值
library(shapviz)
library(fastshap)
pred_cat <- predict_parts_shap(explainer = cat_exp,
                               new_observation = df[-7],
                               type = "shap")
shap_vis <- pred_cat %>% shapviz()
sv_importance(shap_vis, kind = "beeswarm") # 发现得到的蜂群图有点不对，没有体现出“群”的概念

sv_importance(shap_vis, kind = "bar") +
  scale_fill_brewer(palette = "Set2") +
  theme_bw() +
  theme(legend.position = "none") +
  labs(x = "mean(|SHAP valuel) (average impact on model output magnitude)", 
       y = "Feature",
       title = "Importance Ranking of CatBoost based on SHAP Value")
ggsave("SHAP_catboost.pdf",width = 5,height = 4)
  
### 循环作图可视化自变量重要性

model_name <- list(ranger_exp,svm_exp,xgb_exp,kknn_exp,glmnet_exp,log_exp,
                   nb_exp,lda_exp,qda_exp,tree_exp,gbm_exp,nnet_exp,cat_exp)

for (i in model_name){
  print(i[["model"]][["id"]])
  
  # model_parts()函数基于排列组合的重要度来计算变量的重要度
  df_vi <- model_parts(i)
  # head(df_vi)
  
  # 可视化变量重要性，图不好看
  # plot(df_vi, show_boxplots = F)
  
  # 自己做图
  plot_data <- df_vi[df_vi$permutation == 0,]
  plot_data$decrease <- df_vi$dropout_loss[1] - plot_data$dropout_loss 
  plot_data <- plot_data[c(-1,-8),]
  
  ggplot(plot_data,
         aes(x = decrease, 
             y = reorder(variable,decrease), 
             fill = variable)) +
    geom_bar(stat = "identity", width = 0.7) +
    scale_fill_brewer(palette = "Set1") +
    theme_bw() +
    theme(legend.position = "none") +
    labs(x = "One minus AUC loss after permutations", y = "Feature",
         title = paste0("Feature Importance for ", i[["model"]][["id"]]))
  
  ggsave(paste0(i[["model"]][["id"]],"_imp.pdf"),width = 5,height = 4)
}


##### SHAP：CatBoost #####
library(shapviz)
library(permshap)

system.time(
  ps <- permshap(cat_glr, X = df[,-7], bg_X = df)
)
ps
sv <- shapviz(ps)

sv_importance(sv, kind = "beeswarm") # 使用 beeswarm 蜂群图来对数据进行自定义可视化
ggsave("SHAP_catboost_imp.pdf",width = 12,height = 5)

sv_waterfall(sv, row_id = 1)
sv_waterfall(sv, row_id = 2)
ggsave("SHAP_catboost_waterfall.pdf",width = 12,height = 5)

sv_force(sv, row_id = 1)
sv_force(sv, row_id = 2)
ggsave("SHAP_catboost_force.pdf",width = 8,height = 5)

# 进行预测
prediction <- cat_glr$predict(task)
prediction

# 对于分类任务
predicted_labels <- prediction$response  # 类别标签
predicted_probabilities <- prediction$prob  # 类别概率
predicted_probabilities[1:10]

sv_dependence(sv, 
              v = "Grade", # 可以每个自变量都尝试下
              "auto")
