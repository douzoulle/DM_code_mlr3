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
  
  rm(list = ls())
}


##### smote 算法处理阳性事件数不平衡的问题 #####
library(smotefamily)

mul_var <- readRDS("mul_var.Rdata")
data <- read.csv('trainset.csv') %>% 
  .[,c(mul_var,"Distant_Metastasis")]

class(data)
data$Distant_Metastasis = factor(data$Distant_Metastasis)
for (i in names(data)[c(1:6)]){data[,i] <- as.numeric(data[,i])}

task <- TaskClassif$new(id = "example", backend = data, target = "Distant_Metastasis")
task$feature_types$type # 需要为numeric

pop = po("smote")
smotedata = pop$train(list(task))[[1]]$data()

table(data$Distant_Metastasis)
table(smotedata$Distant_Metastasis)
write.csv(smotedata,file="trainset_smote.csv")

