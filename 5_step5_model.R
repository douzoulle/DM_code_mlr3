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
  library(mlr3learners.catboost)
  library(data.table)
  library("magrittr")
  set.seed(2023)
  rm(list = ls())
}

# 读取临床变量排序数据
svm_imp <- read.csv("svm_result.csv") %>% .[,-1]
gbm_imp <- read.csv("gbm_result.csv") %>% .[,-1]
rp_imp <- read.csv("rp_result.csv") %>% .[,-1]
rf_imp <- read.csv("rf_result.csv") %>% .[,-1]
xgb_imp <- read.csv("xgb_result.csv") %>% .[,-1]
cat_imp <- read.csv("cat_result.csv") %>% .[,-1]

# RRA整合排名
library(RobustRankAggreg)

# 产生六个变量列表）
var_list <- list(svm_imp = svm_imp$features, 
                 gbm_imp = gbm_imp$features, 
                 rp_imp = rp_imp$features,
                 rf_imp = rf_imp$features,
                 xgb_imp = xgb_imp$features,
                 cat_imp = cat_imp$features)

# 统计所有临床变量出现的次数
freq = as.data.frame(table(unlist(var_list)))

# 应用RRA算法，对临床变量进行整合排序
ag = aggregateRanks(var_list)

# 添加临床变量出现的次数
ag$Freq = freq[match(ag$Name, freq$Var1),2]
write.csv(ag,"RRA_ag.csv")
saveRDS(ag$Name[1:6],"final_var.rds")

# 读入临床数据
train <- read.csv('trainset_smote.csv') %>% 
  .[,c(ag$Name[1:6], "Distant_Metastasis")]
test <- read.csv('testset.csv') %>% 
  .[,c(ag$Name[1:6], "Distant_Metastasis")]

df <- read.csv('trainset.csv') %>% 
  .[,c(ag$Name[1:6], "Distant_Metastasis")]

##### 相关性热图，数值型变量，ggplot2 #####
cor_data <- cor(df) %>% 
  as.data.frame() %>%
  rownames_to_column("var1") %>% 
  pivot_longer(.,cols = -c("var1"),
               names_to = "var2", 
               values_to = "cor"); cor_data
cor_data$cor <- round(cor_data$cor, 2)

cor_data$var1 <- factor(cor_data$var1,
                        levels = c("Chemotherapy", "Grade", "Radiation", "Tumor_Primary_Site", 
                                   "Regional_Lymph_Nodes", "Surgery_Type", "Distant_Metastasis"))
cor_data$var2 <- factor(cor_data$var2,
                        levels = c("Chemotherapy", "Grade", "Radiation", "Tumor_Primary_Site", 
                                   "Regional_Lymph_Nodes", "Surgery_Type", "Distant_Metastasis"))

pdf("cor_heatmap.pdf",8,6)
ggplot(cor_data, 
       aes(var1, var2)) +
  geom_tile(aes(fill = cor)) +
  geom_text(aes(label = cor), 
            color = "black", size = 5) + 
  scale_fill_gradient2(
    low = 'blue', high = 'red', mid = 'white', limit = c(-1, 1),
    name = paste0("Correlation")) + 
  labs(x = NULL, y = NULL) + 
  theme_bw(base_size = 15)+
  theme(axis.text.x = element_text(size = 12, angle = 45, hjust = 1, color = "black"),
        axis.text.y = element_text(size = 12, color = "black"),
        axis.ticks.y = element_blank(),
        panel.background = element_blank())
dev.off()

##### 模型构建 #####
rm(list = setdiff(ls(), c("train","test")))

# 10折交叉验证
set.seed(2023) # 设置随机种子保证结果可以重复
cv <- rsmp("repeated_cv", folds = 10, repeats = 10) # 内部10次，外部10次

# 创建任务
train$Distant_Metastasis <- as.factor(train$Distant_Metastasis)
test$Distant_Metastasis <- as.factor(test$Distant_Metastasis)

task_train <- as_task_classif(train, target = "Distant_Metastasis", positive = "1")
task_test <- as_task_classif(test, target = "Distant_Metastasis", positive = "1")

# 数据预处理
pbp_prep <- 
  po("scale", scale = F) %>>% # 中心化
  po("removeconstants") # 去掉零方差变量

# 查看训练集任务信息
task_prep <- pbp_prep$clone()$train(task_train)[[1]]
dim(task_train$data())
task_prep$feature_types

# 选择多个模型
{
  # CatBoost
  cat_glr <- as_learner(pbp_prep %>>% lrn("classif.catboost", predict_type="prob")) 
  cat_glr$id <- "CatBoost"
  
  # 决策树
  tree_glr <- as_learner(pbp_prep %>>% lrn("classif.rpart", predict_type="prob")) 
  tree_glr$id <- "decisionTree"
  
  # gbm
  gbm_glr <- as_learner(pbp_prep %>>% lrn("classif.svm", predict_type="prob")) 
  gbm_glr$id <- "GBM"

  # 带正则化的广义线性模型
  glmnet_glr <- as_learner(pbp_prep %>>% lrn("classif.glmnet", predict_type="prob")) 
  glmnet_glr$id <- "glmnet"
  
  # k近邻
  kknn_glr <- as_learner(pbp_prep %>>% lrn("classif.kknn", predict_type="prob")) 
  kknn_glr$id <- "kknn"
  
  # lda
  lda_glr <- as_learner(pbp_prep %>>% lrn("classif.lda", predict_type="prob")) 
  lda_glr$id <- "lda"
  
  # 逻辑回归
  log_glr <-as_learner(pbp_prep %>>% lrn("classif.log_reg", predict_type="prob")) 
  log_glr$id <- "logistic"
  
  # 朴素贝叶斯
  nb_glr <-as_learner(pbp_prep %>>% lrn("classif.naive_bayes", predict_type="prob")) 
  nb_glr$id <- "NaiveBayes"
  
  # 神经网络
  nnet_glr <- as_learner(pbp_prep %>>% lrn("classif.nnet", predict_type="prob")) 
  nnet_glr$id <- "nnet"
  
  # qda
  qda_glr <- as_learner(pbp_prep %>>% lrn("classif.qda", predict_type="prob")) 
  qda_glr$id <- "qda"
  
  # 随机森林
  rf_glr <- as_learner(pbp_prep %>>% lrn("classif.ranger", predict_type="prob")) 
  rf_glr$id <- "randomForest"
  
  # svm
  svm_glr <- as_learner(pbp_prep %>>% lrn("classif.svm", predict_type="prob")) 
  svm_glr$id <- "SVM"
  
  # xgb
  xgb_glr <- as_learner(pbp_prep %>>% lrn("classif.xgboost", predict_type="prob")) 
  xgb_glr$id <- "XGboost"
}

# 创建benchmark对象
design = benchmark_grid(
  tasks = task_train,
  learners = list(cat_glr,tree_glr,gbm_glr,glmnet_glr,kknn_glr,lda_glr,
                  log_glr,nb_glr,nnet_glr,qda_glr,rf_glr,svm_glr,xgb_glr),
  resamplings = cv) # 十折交叉验证

# 加速
library(future)
plan("multisession", workers = 8)

# 减少屏幕输出
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")

# 执行
bmr = benchmark(design, store_models = T); bmr

# 获得每一个模型的评价指标
mlr_measures
measures <- msrs(c("classif.auc", "classif.acc", "classif.ce", "classif.precision", "classif.prauc",
                   "classif.sensitivity", "classif.specificity", "classif.bbrier")) # recall和sensitivity其实一样
tab = bmr$aggregate(measures); tab

# 模型排序
library(mlr3benchmark)
bma = as_benchmark_aggr(bmr, measures = msr("classif.auc"))
bma$rank_data(minimize = FALSE) # 可以看见，catboost效果最好

# 错误率可视化
autoplot(bmr) + 
  theme(axis.text.x = element_text(angle = 45))

# ROC曲线
autoplot(bmr, type = "roc")

# PRC曲线
autoplot(bmr, type = "prc")

# 比较auc值
autoplot(bmr, measure = msr("classif.auc"))

##### 十折交叉验证auc变化曲线 #####

# 内10外10的交叉验证
auc_rf = tab[learner_id == "randomForest"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_rf <- data.table(learner_id = rep("randomForest", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_rf$classif.auc[i] <- mean(auc_rf$classif.auc[start_index:end_index])
}

auc_svm = tab[learner_id == "SVM"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_svm <- data.table(learner_id = rep("SVM", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_svm$classif.auc[i] <- mean(auc_svm$classif.auc[start_index:end_index])
}

auc_xgb = tab[learner_id == "XGboost"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_xgb <- data.table(learner_id = rep("XGboost", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_xgb$classif.auc[i] <- mean(auc_xgb$classif.auc[start_index:end_index])
}

auc_kknn = tab[learner_id == "kknn"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_kknn <- data.table(learner_id = rep("kknn", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_kknn$classif.auc[i] <- mean(auc_kknn$classif.auc[start_index:end_index])
}

auc_glmnet = tab[learner_id == "glmnet"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_glmnet <- data.table(learner_id = rep("glmnet", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_glmnet$classif.auc[i] <- mean(auc_glmnet$classif.auc[start_index:end_index])
}

auc_log = tab[learner_id == "logistic"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_log <- data.table(learner_id = rep("logistic", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_log$classif.auc[i] <- mean(auc_log$classif.auc[start_index:end_index])
}

auc_nb = tab[learner_id == "NaiveBayes"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_nb <- data.table(learner_id = rep("NaiveBayes", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_nb$classif.auc[i] <- mean(auc_nb$classif.auc[start_index:end_index])
}

auc_lda = tab[learner_id == "lda"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_lda <- data.table(learner_id = rep("lda", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_lda$classif.auc[i] <- mean(auc_lda$classif.auc[start_index:end_index])
}

auc_qda = tab[learner_id == "qda"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_qda <- data.table(learner_id = rep("qda", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_qda$classif.auc[i] <- mean(auc_qda$classif.auc[start_index:end_index])
}

auc_tree = tab[learner_id == "decisionTree"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_tree <- data.table(learner_id = rep("decisionTree", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_tree$classif.auc[i] <- mean(auc_tree$classif.auc[start_index:end_index])
}

auc_gbm = tab[learner_id == "GBM"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_gbm <- data.table(learner_id = rep("GBM", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_gbm$classif.auc[i] <- mean(auc_gbm$classif.auc[start_index:end_index])
}

auc_nnet = tab[learner_id == "nnet"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_nnet <- data.table(learner_id = rep("nnet", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_nnet$classif.auc[i] <- mean(auc_nnet$classif.auc[start_index:end_index])
}

auc_cat = tab[learner_id == "CatBoost"]$resample_result[[1]]$score(measures)[,c(4,9)]
mean_cat <- data.table(learner_id = rep("CatBoost", 10))
for (i in 1:10) {
  start_index <- (i - 1) * 10 + 1
  end_index <- i * 10
  mean_cat$classif.auc[i] <- mean(auc_cat$classif.auc[start_index:end_index])
}

long_data <- rbind(mean_rf, mean_svm, mean_xgb, mean_kknn, mean_glmnet, mean_log, 
                   mean_nb, mean_lda, mean_qda, mean_tree, mean_gbm, mean_nnet, mean_cat) %>% 
  mutate(iter = rep(1:10, 13))
long_data$learner_id <- factor(long_data$learner_id,
                               levels =c("randomForest", "decisionTree", "GBM", "glmnet", "kknn", "lda", 
                                         "logistic", "NaiveBayes", "nnet", "qda", "CatBoost", "SVM", "XGboost"))

pdf("auc_10k_train.pdf",7,4)
ggplot(data = long_data,
       aes(x = iter, y = classif.auc)) +
  geom_line(aes(group = learner_id, color = learner_id), size = 0.4) +
  geom_point(data = long_data[long_data$iter %% 1 == 0, ], aes(color = learner_id), size = 0.9) +
  scale_x_continuous(breaks = seq(min(long_data$iter)-1, max(long_data$iter), by = 1)) +
  scale_y_continuous(breaks = seq(0.75, 0.9, by = 0.05), limits = c(0.75, 0.9)) +
  theme(panel.background = element_blank(),
        axis.line = element_line(),
        legend.position = "bottom",
        axis.text.x = element_text(angle = 60, hjust = 1)) +
  guides(shape = guide_legend(title.position = "top"))
dev.off()

##### 训练集囊括所有评价指标的热图 #####
data_long <- tab %>% 
  .[, c(4,7:10,12:14)] %>% 
  pivot_longer(!learner_id, names_to = "measure", values_to = "value") 

data_long$measure <-
  data_long$measure |> 
  str_replace_all("classif.","")

data_long$value <- round(data_long$value,3)
data_long$learner_id <- factor(data_long$learner_id,
                               levels = c("XGboost", "SVM", "glmnet", "qda", "nnet", "NaiveBayes", "logistic",
                                           "lda", "kknn", "CatBoost", "GBM", "decisionTree", "randomForest"))
                                          
pdf("heatmap_train.pdf",9,6)
ggplot(data_long, 
       aes(measure, learner_id)) + 
  geom_tile(
    aes(fill = value), colour = "grey", size = 1)+
  scale_fill_gradient2(
    low = "steelblue", mid = "white", high = "darkred", midpoint = 0.5) + 
  geom_text(
    aes(label = value), col = "black", size = 5) +
  theme_minimal() + # 不要背景
  theme(axis.title.x = element_blank(), # 去掉title
        axis.ticks.x = element_blank(), # 去掉x轴
        axis.title.y = element_blank(), # 去掉y轴
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14, face = "bold"), # 调整x轴文字，字体加粗
        axis.text.y = element_text(size = 14, face = "bold")) + # 调整y轴文字
  labs(fill = paste0("benchmark")) + # 修改legend内容
  scale_x_discrete(position = "bottom") # 将X轴放置在最上面
dev.off()

### 训练集校正曲线
rf_glr$train(task_train)
prediction <- as.data.table(rf_glr$predict(task_train))
head(prediction)

# 散点图
calibration_df <- prediction %>% 
  mutate(yes = if_else(truth == "1", 1, 0),
         pred_rnd = round(prob.1, 2)) %>% 
  dplyr::group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(prob.1),
                   mean_obs = mean(yes),
                   n = n())

ggplot(calibration_df, 
       aes(mean_pred, mean_obs))+ 
  geom_point(aes(size = n), alpha = 0.5)+
  scale_color_lancet()+
  geom_abline(linetype = "dashed")+
  labs(x = "Predicted Probability", y = "Observed Probability")+
  theme_minimal()

# 折线图
cali_df <- prediction %>% 
  arrange(prob.1) %>% 
  mutate(yes = if_else(truth == "1", 1, 0),
         group = c(rep(1:100, each = 11), rep(101,68))) %>% 
  dplyr::group_by(group) %>% 
  dplyr::summarise(mean_pred = mean(prob.1),
                   mean_obs = mean(yes))

ggplot(cali_df, aes(mean_pred, mean_obs))+ 
  geom_line(size=1)+
  labs(x = "Predicted Probability", y = "Observed Probability")+
  theme_minimal()

##### 模型评估 #####
#模型评估主要从三方面开始，分别是区分度、校准度、临床决策曲线DCA，而区分度、DCA是评价logistic的可视化结果

# 选择表现最好的模型：随机森林，进行训练
cat_glr$train(task_train)

# 测试集测试
pred_rf <- cat_glr$predict(task_test)
head(as.data.table(cat_glr$predict(task_test)))

# 混淆矩阵
pred_rf$confusion
autoplot(pred_rf)

# 查看多个模型评价指标
pred_rf$score(measures)

# ROC曲线
autoplot(pred_rf, type = "roc")

##### 验证集囊括所有评价指标的热图 #####

# 输出每个模型在验证集中训练的结果
{
rf_glr$train(task_train)
pred_rf <- rf_glr$predict(task_test)
pred_rf_res <- pred_rf$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "randomForest")

svm_glr$train(task_train)
pred_svm <- svm_glr$predict(task_test)
pred_svm_res <- pred_svm$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "SVM")

xgb_glr$train(task_train)
pred_xgb <- xgb_glr$predict(task_test)
pred_xgb_res <- pred_xgb$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "XGboost")

kknn_glr$train(task_train)
pred_kknn <- kknn_glr$predict(task_test)
pred_kknn_res <- pred_kknn$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "kknn")

glmnet_glr$train(task_train)
pred_glmnet <- glmnet_glr$predict(task_test)
pred_glmnet_res <- pred_glmnet$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "glmnet")

log_glr$train(task_train)
pred_log <- log_glr$predict(task_test)
pred_log_res <- pred_log$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "logistic")

nb_glr$train(task_train)
pred_nb <- nb_glr$predict(task_test)
pred_nb_res <- pred_nb$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "NaiveBayes")

lda_glr$train(task_train)
pred_lda <- lda_glr$predict(task_test)
pred_lda_res <- pred_lda$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "lda")

qda_glr$train(task_train)
pred_qda <- qda_glr$predict(task_test)
pred_qda_res <- pred_qda$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "qda")

tree_glr$train(task_train)
pred_tree <- tree_glr$predict(task_test)
pred_tree_res <- pred_tree$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "decisionTree")

gbm_glr$train(task_train)
pred_gbm <- gbm_glr$predict(task_test)
pred_gbm_res <- pred_gbm$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "GBM")

nnet_glr$train(task_train)
pred_nnet <- nnet_glr$predict(task_test)
pred_nnet_res <- pred_nnet$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "nnet")

cat_glr$train(task_train)
pred_cat <- cat_glr$predict(task_test)
pred_cat_res <- pred_cat$score(measures) |> 
  as.data.frame() |> 
  t() |>
  as.data.frame() |>
  dplyr::mutate(learner_id = "CatBoost")
}

# 合并
tab_test <- rbind(pred_rf_res, pred_svm_res, pred_xgb_res, pred_kknn_res, 
                  pred_glmnet_res, pred_log_res, pred_nb_res, pred_lda_res,
                  pred_qda_res, pred_tree_res, pred_gbm_res, pred_nnet_res, pred_cat_res)

# 转换为长数据
data_long_test <- tab_test[,c(1:4,6:9)] %>% 
  pivot_longer(!learner_id, names_to = "measure", values_to = "value") 

data_long_test$measure <-
  data_long_test$measure |> 
  str_replace_all("classif.","")

data_long_test$value <- round(data_long_test$value, 3)
data_long_test$learner_id <- factor(data_long_test$learner_id,
                                    levels = c("XGboost", "SVM", "CatBoost", "qda", "nnet", "NaiveBayes", "logistic",
                                               "lda", "kknn", "glmnet", "GBM", "decisionTree", "randomForest"))

pdf("heatmap_test.pdf",9,6)
ggplot(data_long_test, 
       aes(measure, learner_id)) + 
  geom_tile(
    aes(fill = value), colour = "grey", size = 1)+
  scale_fill_gradient2(
    low = "steelblue", mid = "white", high = "darkred", midpoint = 0.5) + 
  geom_text(
    aes(label = value), col = "black", size = 5) +
  theme_minimal() + # 不要背景
  theme(axis.title.x = element_blank(), # 去掉title
        axis.ticks.x = element_blank(), # 去掉x轴
        axis.title.y = element_blank(), # 去掉y轴
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14, face = "bold"), # 调整x轴文字，字体加粗
        axis.text.y = element_text(size = 14, face = "bold")) + # 调整y轴文字
  labs(fill = paste0("benchmark")) + # 修改legend内容
  scale_x_discrete(position = "bottom") # 将X轴放置在最上面
dev.off()

##### ex #####

# 分出外部验证集
ex <- read.csv('ex.csv') %>% 
  .[,c("Chemotherapy", "Grade", "Radiation", "Regional_Lymph_Nodes", 
       "Surgery_Type", "Tumor_Primary_Site", "Distant_Metastasis")]
task_test <- as_task_classif(ex,
                             target = "Distant_Metastasis",
                             positive = "1")

# 测试集测试
pred_rf <- cat_glr$predict(task_test)
head(as.data.table(cat_glr$predict(task_test)))

# 混淆矩阵
pred_rf$confusion
autoplot(pred_rf)

# 输出每个模型在外部验证集中训练的结果
{
  rf_glr$train(task_train)
  pred_rf <- rf_glr$predict(task_test)
  pred_rf_res <- pred_rf$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "randomForest")
  
  svm_glr$train(task_train)
  pred_svm <- svm_glr$predict(task_test)
  pred_svm_res <- pred_svm$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "SVM")
  
  xgb_glr$train(task_train)
  pred_xgb <- xgb_glr$predict(task_test)
  pred_xgb_res <- pred_xgb$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "XGboost")
  
  kknn_glr$train(task_train)
  pred_kknn <- kknn_glr$predict(task_test)
  pred_kknn_res <- pred_kknn$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "kknn")
  
  glmnet_glr$train(task_train)
  pred_glmnet <- glmnet_glr$predict(task_test)
  pred_glmnet_res <- pred_glmnet$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "glmnet")
  
  log_glr$train(task_train)
  pred_log <- log_glr$predict(task_test)
  pred_log_res <- pred_log$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "logistic")
  
  nb_glr$train(task_train)
  pred_nb <- nb_glr$predict(task_test)
  pred_nb_res <- pred_nb$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "NaiveBayes")
  
  lda_glr$train(task_train)
  pred_lda <- lda_glr$predict(task_test)
  pred_lda_res <- pred_lda$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "lda")
  
  qda_glr$train(task_train)
  pred_qda <- qda_glr$predict(task_test)
  pred_qda_res <- pred_qda$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "qda")
  
  tree_glr$train(task_train)
  pred_tree <- tree_glr$predict(task_test)
  pred_tree_res <- pred_tree$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "decisionTree")
  
  gbm_glr$train(task_train)
  pred_gbm <- gbm_glr$predict(task_test)
  pred_gbm_res <- pred_gbm$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "GBM")
  
  nnet_glr$train(task_train)
  pred_nnet <- nnet_glr$predict(task_test)
  pred_nnet_res <- pred_nnet$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "nnet")
  
  cat_glr$train(task_train)
  pred_cat <- cat_glr$predict(task_test)
  pred_cat_res <- pred_cat$score(measures) |> 
    as.data.frame() |> 
    t() |>
    as.data.frame() |>
    dplyr::mutate(learner_id = "CatBoost")
}

# 合并
tab_test <- rbind(pred_rf_res, pred_svm_res, pred_xgb_res, pred_kknn_res, 
                  pred_glmnet_res, pred_log_res, pred_nb_res, pred_lda_res,
                  pred_qda_res, pred_tree_res, pred_gbm_res, pred_nnet_res, pred_cat_res)

# 转换为长数据
data_long_test <- tab_test[,c(1:4,6:9)] %>% 
  pivot_longer(!learner_id, names_to = "measure", values_to = "value") 

data_long_test$measure <-
  data_long_test$measure |> 
  str_replace_all("classif.","")

data_long_test$value <- round(data_long_test$value, 3)
data_long_test$learner_id <- factor(data_long_test$learner_id,
                                    levels = c("XGboost", "SVM", "randomForest", "qda", "nnet", 
                                               "NaiveBayes", "logistic", "lda", "kknn", "glmnet",
                                               "GBM", "decisionTree", "CatBoost"))

pdf("heatmap_ex.pdf",9,6)
ggplot(data_long_test, 
       aes(measure, learner_id)) + 
  geom_tile(
    aes(fill = value), colour = "grey", size = 1)+
  scale_fill_gradient2(
    low = "steelblue", mid = "white", high = "darkred", midpoint = 0.5) + 
  geom_text(
    aes(label = value), col = "black", size = 5) +
  theme_minimal() + # 不要背景
  theme(axis.title.x = element_blank(), # 去掉title
        axis.ticks.x = element_blank(), # 去掉x轴
        axis.title.y = element_blank(), # 去掉y轴
        axis.text.x = element_text(angle = 45, hjust = 1, size = 14, face = "bold"), # 调整x轴文字，字体加粗
        axis.text.y = element_text(size = 14, face = "bold")) + # 调整y轴文字
  labs(fill = paste0("benchmark")) + # 修改legend内容
  scale_x_discrete(position = "bottom") # 将X轴放置在最上面
dev.off()

