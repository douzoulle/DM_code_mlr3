{
  library(tidyverse) # 数据处理
  library(ggplot2) # 数据可视化
  library(mice) # 缺失值填补
  library(missForest) # 实现随机森林插补缺失值
  library(DMwR2) # 实现KNN插补缺失值
  library(relaimpo) # 筛选重要变量
  library(leaps) # 最优子集
  library(glmnet) # 执行正则化技术
  library(lmtest) # 模型检测
  library(gvlma) # 模型检测
  library(car) # 模型检测
  library(ggDCA) # 绘制DCA
  library(ggsci)
  library(mlr3) # 主体包
  library(mlr3viz) # 执行可视化功能
  library(mlr3learners) # 提供额外学习器
  library(mlr3verse) # 扩展包
  library(mlr3tuning) # 调整参数
  library(data.table)
  library("magrittr")
  library(viridisLite)
  library(mlr3misc)
  set.seed(123)
  rm(list = ls())
}

mul_var <- readRDS("mul_var.Rdata")

##### GBM #####
rm(list = setdiff(ls(),"mul_var"))

df <- read.csv('trainset.csv') %>% 
  .[,c(mul_var,"Distant_Metastasis")]
for (i in names(df)[c(1:7)]){df[,i] <- as.factor(df[,i])}

library(mlr3verse)
library(mlr3extralearners)

task = TaskClassif$new(id = "nb",
                       backend = df,
                       target = "Distant_Metastasis",
                       positive = "1")

optimizer = fs("rfe",
               n_features = 1,
               feature_number = 1,
               aggregation = "rank")

learner = lrn("classif.gbm",
              distribution = "bernoulli",
              predict_type = "prob")

instance = fsi(task = task,
               learner = learner,
               resampling = rsmp("cv", folds = 10),
               measures = msr("classif.auc"),
               terminator = trm("none"))

optimizer$optimize(instance)

library(viridisLite)
library(mlr3misc)

data_gbm = as.data.table(instance$archive)
data_gbm[, n:= map_int(importance, length)]

num <- length(instance$result_feature_set)
auc <- data_gbm$classif.auc[data_gbm$n == num]

pdf("fs_gbm.pdf",6,6)
ggplot(data_gbm, 
       aes(x = n, y = classif.auc)) +
  geom_line(
    color = viridis(1, begin = 0.5),
    linewidth = 1) +
  geom_point(
    fill = viridis(1, begin = 0.5),
    shape = 21,
    size = 3,
    stroke = 0.5,
    alpha = 0.8) +
  geom_vline(
    xintercept = num,
    linetype = "dashed",
    color = "gray") +
  geom_text(
    aes(x = num, y = 0.69, label = paste(c("n ="), num)),
    color = "black", hjust = 0) +
  geom_text(
    aes(x = num, y = auc + 0.003, label = paste(c("auc ="), round(auc,3))),
    color = "black", hjust = 0) +
  xlab("Number of Features") +
  ggtitle("Gradient Boosting Machine") +
  scale_x_reverse() +
  theme_minimal()
dev.off()

# 查看最终保留的特征
instance$result_feature_set
saveRDS(instance$result_feature_set, "gbm_var.Rdata")

gbm_result <- as.data.table(instance$archive)[, list(features, classif.auc, importance)]
head(gbm_result)

classif.auc <- list()
for (i in (1:nrow(gbm_result))){
  classif.auc[[i]] = rep(gbm_result$classif.auc[i], 
                         each = length(gbm_result$features[[i]]))
}

gbm <- data.frame(features = unlist(gbm_result$features),
                  classif.auc = unlist(classif.auc),
                  importance = unlist(gbm_result$importance))
gbm <- gbm %>% 
  dplyr::filter(classif.auc == auc)

write.csv(gbm,"gbm_result.csv")

##### SVM #####
set.seed(2023)
rm(list = setdiff(ls(),"mul_var"))

df <- read.csv('trainset.csv') %>% 
  .[,c(mul_var,"Distant_Metastasis")]

df1 <- scale(df[,-7]) %>% 
  as.data.frame()
df1$Distant_Metastasis <- as.factor(df$Distant_Metastasis)

task = TaskClassif$new(id = "nb",
                       backend = df1,
                       target = "Distant_Metastasis",
                       positive = "1")

learner = lrn("classif.svm",
              type = "C-classification",
              kernel = "linear",
              predict_type = "prob")

learner$properties

instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 10),
  measures = msr("classif.auc"),
  terminator = trm("none"),
  callback = clbk("mlr3fselect.svm_rfe"))

optimizer = fs("rfe",
               n_features = 1,
               feature_number = 1,
               aggregation = "rank")

optimizer$optimize(instance)

data_svm = as.data.table(instance$archive)
data_svm[, n:= map_int(importance, length)]

num <- length(instance$result_feature_set)
auc <- data_svm$classif.auc[data_svm$n == num]

pdf("fs_svm.pdf",6,6)
ggplot(data_svm, 
       aes(x = n, y = classif.auc)) +
  geom_line(
    color = viridis(1, begin = 0.5),
    linewidth = 1) +
  geom_point(
    fill = viridis(1, begin = 0.5),
    shape = 21,
    size = 3,
    stroke = 0.5,
    alpha = 0.8) +
  geom_vline(
    xintercept = num,
    linetype = "dashed",
    color = "gray") +
  geom_text(
    aes(x = num, y = 0.69, label = paste(c("n ="), num)),
    color = "black", hjust = 0) +
  geom_text(
    aes(x = num, y = auc + 0.003, label = paste(c("auc ="), round(auc,3))),
    color = "black", hjust = 0) +
  xlab("Number of Features") +
  ggtitle("Support Vector Machines") +
  scale_x_reverse() +
  theme_minimal()
dev.off()

# 查看最终保留的特征
instance$result_feature_set
saveRDS(instance$result_feature_set, "svm_var.Rdata")

svm_result <- as.data.table(instance$archive)[, list(features, classif.auc, importance)]
head(svm_result)

classif.auc <- list()
for (i in (1:nrow(svm_result))){
  classif.auc[[i]] = rep(svm_result$classif.auc[i], 
                         each = length(svm_result$features[[i]]))
}

svm <- data.frame(features = unlist(svm_result$features),
                  classif.auc = unlist(classif.auc),
                  importance = unlist(svm_result$importance))
svm <- svm %>% 
  dplyr::filter(classif.auc == auc)

write.csv(svm,"svm_result.csv")

##### ranger #####
rm(list = setdiff(ls(),"mul_var"))

df <- read.csv('trainset.csv') %>% 
  .[,c(mul_var,"Distant_Metastasis")]
for (i in names(df)[c(1:7)]){df[,i] <- as.factor(df[,i])}

library(mlr3verse)
library(mlr3extralearners)

task = TaskClassif$new(id = "nb",
                       backend = df,
                       target = "Distant_Metastasis",
                       positive = "1")

optimizer = fs("rfe",
               n_features = 1,
               feature_number = 1,
               aggregation = "rank")

learner = lrn("classif.ranger",
              predict_type = "prob",
              importance = "permutation")
LearnerClassifRanger$importance()

instance = fsi(task = task,
               learner = learner,
               resampling = rsmp("cv", folds = 10),
               measures = msr("classif.auc"),
               terminator = trm("none"))

optimizer$optimize(instance)

library(viridisLite)
library(mlr3misc)

data_rf = as.data.table(instance$archive)
data_rf[, n:= map_int(importance, length)]

num <- length(instance$result_feature_set)
auc <- data_rf$classif.auc[data_rf$n == num]

pdf("fs_rf.pdf",6,6)
ggplot(data_rf, 
       aes(x = n, y = classif.auc)) +
  geom_line(
    color = viridis(1, begin = 0.5),
    linewidth = 1) +
  geom_point(
    fill = viridis(1, begin = 0.5),
    shape = 21,
    size = 3,
    stroke = 0.5,
    alpha = 0.8) +
  geom_vline(
    xintercept = num,
    linetype = "dashed",
    color = "gray") +
  geom_text(
    aes(x = num, y = 0.7, label = paste(c("n ="), num)),
    color = "black", hjust = 0) +
  geom_text(
    aes(x = num, y = auc + 0.003, label = paste(c("auc ="), round(auc,3))),
    color = "black", hjust = 0) +
  xlab("Number of Features") +
  ggtitle("Random Forest") +
  scale_x_reverse() +
  theme_minimal()
dev.off()

# 查看最终保留的特征
instance$result_feature_set
saveRDS(instance$result_feature_set, "rf_var.Rdata")

rf_result <- as.data.table(instance$archive)[, list(features, classif.auc, importance)]
head(rf_result)

classif.auc <- list()
for (i in (1:nrow(rf_result))){
  classif.auc[[i]] = rep(rf_result$classif.auc[i], 
                         each = length(rf_result$features[[i]]))
}

rf <- data.frame(features = unlist(rf_result$features),
                 classif.auc = unlist(classif.auc),
                 importance = unlist(rf_result$importance))
rf <- rf %>% 
  dplyr::filter(classif.auc == auc)

write.csv(rf,"rf_result.csv")

##### rpart #####
rm(list = setdiff(ls(),"mul_var"))

df <- read.csv('trainset.csv') %>% 
  .[,c(mul_var,"Distant_Metastasis")]
for (i in names(df)[c(1:7)]){df[,i] <- as.factor(df[,i])}

library(mlr3verse)
library(mlr3extralearners)

task = TaskClassif$new(id = "nb",
                       backend = df,
                       target = "Distant_Metastasis",
                       positive = "1")

optimizer = fs("rfe",
               n_features = 1,
               feature_number = 1,
               aggregation = "rank")

learner_rpart = lrn("classif.rpart",
                    predict_type = "prob")

instance = fsi(task = task,
               learner = learner_rpart,
               resampling = rsmp("cv", folds = 10),
               measures = msr("classif.auc"),
               terminator = trm("none"))

optimizer$optimize(instance)

library(viridisLite)
library(mlr3misc)

data_rp = as.data.table(instance$archive)
data_rp[, n:= map_int(importance, length)]

num <- length(instance$result_feature_set)
auc <- data_rp$classif.auc[data_rp$n == num]

pdf("fs_rp.pdf",6,6)
ggplot(data_rp, 
       aes(x = n, y = classif.auc)) +
  geom_line(
    color = viridis(1, begin = 0.5),
    linewidth = 1) +
  geom_point(
    fill = viridis(1, begin = 0.5),
    shape = 21,
    size = 3,
    stroke = 0.5,
    alpha = 0.8) +
  geom_vline(
    xintercept = num,
    linetype = "dashed",
    color = "gray") +
  geom_text(
    aes(x = num, y = 0.7, label = paste(c("n ="), num)),
    color = "black", hjust = 0) +
  geom_text(
    aes(x = num, y = auc + 0.003, label = paste(c("auc ="), round(auc,3))),
    color = "black", hjust = 0) +
  xlab("Number of Features") +
  ggtitle("Decision Tree") +
  scale_x_reverse() +
  theme_minimal()
dev.off()

# 查看最终保留的特征
instance$result_feature_set
saveRDS(instance$result_feature_set, "rp_var.Rdata")

rp_result <- as.data.table(instance$archive)[, list(features, classif.auc, importance)]
head(rp_result)

classif.auc <- list()
for (i in (1:nrow(rp_result))){
  classif.auc[[i]] = rep(rp_result$classif.auc[i], 
                         each = length(rp_result$features[[i]]))
}

rp <- data.frame(features = unlist(rp_result$features),
                 classif.auc = unlist(classif.auc),
                 importance = unlist(rp_result$importance))
rp <- rp %>% 
  dplyr::filter(classif.auc == auc) %>% 
  .[1:num,]

write.csv(rp,"rp_result.csv")

##### XGB #####
rm(list = setdiff(ls(),"mul_var"))

df <- read.csv('trainset.csv') %>% 
  .[,c(mul_var,"Distant_Metastasis")]
for (i in names(df)[c(7)]){df[,i] <- as.factor(df[,i])}

task = TaskClassif$new(id = "nb",
                       backend = df,
                       target = "Distant_Metastasis",
                       positive = "1")

learner = lrn("classif.xgboost",
              predict_type = "prob")

learner$properties
lrn("classif.xgboost")$param_set$levels$importance

instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 10),
  measures = msr("classif.auc"),
  terminator = trm("none"))

optimizer = fs("rfe",
               n_features = 1,
               feature_number = 1,
               aggregation = "rank")

optimizer$optimize(instance)

data_xgb = as.data.table(instance$archive)
data_xgb[, n:= map_int(importance, length)]

num <- length(instance$result_feature_set)
num <- 5
auc <- data_xgb$classif.auc[data_xgb$n == num]

pdf("fs_xgb.pdf",6,6)
ggplot(data_xgb, 
       aes(x = n, y = classif.auc)) +
  geom_line(
    color = viridis(1, begin = 0.5),
    linewidth = 1) +
  geom_point(
    fill = viridis(1, begin = 0.5),
    shape = 21,
    size = 3,
    stroke = 0.5,
    alpha = 0.8) +
  geom_vline(
    xintercept = num,
    linetype = "dashed",
    color = "gray") +
  geom_text(
    aes(x = num, y = 0.69, label = paste(c("n ="), num)),
    color = "black", hjust = 0) +
  geom_text(
    aes(x = num, y = auc + 0.003, label = paste(c("auc ="), round(auc,3))),
    color = "black", hjust = 0) +
  xlab("Number of Features") +
  ggtitle("eXtreme Gradient Boosting") +
  scale_x_reverse() +
  theme_minimal()
dev.off()

# 查看最终保留的特征
# instance$result_feature_set
tmp <- data_xgb[data_xgb$n == num] %>% 
  as.data.frame()
var <- names(tmp)[tmp[1, 1:7] == TRUE]; var

# saveRDS(instance$result_feature_set, "xgb_var.Rdata")
saveRDS(var[1:num], "xgb_var.Rdata")

xgb_result <- as.data.table(instance$archive)[, list(features, classif.auc, importance)]
head(xgb_result)

classif.auc <- list()
for (i in (1:nrow(xgb_result))){
  classif.auc[[i]] = rep(xgb_result$classif.auc[i], 
                         each = length(xgb_result$features[[i]]))
}

xgb <- data.frame(features = unlist(xgb_result$features),
                  classif.auc = unlist(classif.auc),
                  importance = c(unlist(xgb_result$importance),"tmp"))
xgb <- xgb %>% 
  dplyr::filter(classif.auc == auc)

write.csv(xgb,"xgb_result.csv")

##### catboost #####
rm(list = setdiff(ls(),"mul_var"))

df <- read.csv('trainset.csv') %>% 
  .[,c(mul_var,"Distant_Metastasis")]
for (i in names(df)[c(7)]){df[,i] <- as.factor(df[,i])}

# 作者用的catboost算法，而mlr3集成的算法中还没有catboost，所以我们需要让catboost继承到mlr3的框架里去；下面这个操作只是让catboost具备mlr3的父类特征，并没有真正地成为可以操作的对象。
# remotes::install_github("mlr3learners/mlr3learners.catboost")
library(mlr3learners.catboost)

# 接下来，需要做下面的操作，先下载一个 catboost 的版本，根据自己电脑的操作系统，下载相应的压缩包，解压后指定文件路径
# setwd("./catboost")
# devtools::build()
# devtools::install()
# setwd("D:/seer_NB_ML")

learner = mlr3learners.catboost::LearnerClassifCatboost  # 指定学习器名称
generator <- learner$new() # 查看一下 generator，是不是一个 R6 对象 
generator

learner = lrn("classif.catboost",
              predict_type = "prob",
              l2_leaf_reg = 5,
              learning_rate = 0.05) # 添加 catboost 为实例化的学习器

graph = po("encode") %>>% learner
plot(graph, html = FALSE)
graphlearner = as_learner(graph)

learner$properties
lrn("classif.xgboost")$param_set$levels$importance

task = TaskClassif$new(id = "nb",
                       backend = df,
                       target = "Distant_Metastasis",
                       positive = "1")

instance = fsi(
  task = task,
  learner = graphlearner,
  resampling = rsmp("cv", folds = 10),
  measures = msr("classif.auc"),
  terminator = trm("none"))

optimizer = fs("rfe",
               n_features = 1,
               feature_number = 1,
               aggregation = "rank")

optimizer$optimize(instance)

data_cat = as.data.table(instance$archive)
data_cat[, n:= map_int(importance, length)]

num <- length(instance$result_feature_set)
auc <- data_cat$classif.auc[data_cat$n == num]

pdf("fs_cat.pdf",6,6)
ggplot(data_cat, 
       aes(x = n, y = classif.auc)) +
  geom_line(
    color = viridis(1, begin = 0.5),
    linewidth = 1) +
  geom_point(
    fill = viridis(1, begin = 0.5),
    shape = 21,
    size = 3,
    stroke = 0.5,
    alpha = 0.8) +
  geom_vline(
    xintercept = num,
    linetype = "dashed",
    color = "gray") +
  geom_text(
    aes(x = num, y = 0.69, label = paste(c("n ="), num)),
    color = "black", hjust = 0) +
  geom_text(
    aes(x = num, y = auc + 0.003, label = paste(c("auc ="), round(auc,3))),
    color = "black", hjust = 0) +
  xlab("Number of Features") +
  ggtitle("CatBoost") +
  scale_x_reverse() +
  theme_minimal()
dev.off()

# 查看最终保留的特征
instance$result_feature_set
saveRDS(instance$result_feature_set, "cat_var.Rdata")

cat_result <- as.data.table(instance$archive)[, list(features, classif.auc, importance)]
head(cat_result)

classif.auc <- list()
for (i in (1:nrow(cat_result))){
  classif.auc[[i]] = rep(cat_result$classif.auc[i], 
                         each = length(cat_result$features[[i]]))
}

cat <- data.frame(features = unlist(cat_result$features),
                  classif.auc = unlist(classif.auc),
                  importance = unlist(cat_result$importance))
cat <- cat %>% 
  dplyr::filter(classif.auc == auc)

write.csv(cat,"cat_result.csv")
