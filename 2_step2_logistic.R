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

##### 探索数据（EDA）#####
# EDA这个步骤的本质就是加深我们对数据的理解，因为只有对数据足够理解，能帮助我们完成下面三个任务
# 1.特征选择
# 2.模型解释
# 3.数据的额外操作（算法选择、可视化等等）

df <- read.csv('trainset.csv') %>% 
  .[,c(2, 4, 6:17, 20:25)]
for (i in names(df)[c(1:20)]){df[,i] <- as.factor(df[,i])}
str(df) # 获取数据集基本信息

skimr::skim(df) # 获取数据集报告

df %>% group_by(Distant_Metastasis) %>% 
  summarise(mean = mean(Age_Continuous)) # 记忆这种代码书写形式，可以依据分组求某个连续型预测变量的信息

summary(df) # 获取数据集的基本信息（数据方面）

table(complete.cases(df)) # 查找缺失值（TRUE代表没有缺失值）
table(is.na(df)) # 查找缺失值

##### 特征选择 #####
# 特征选择在mlr3中介绍了很多，但是假如你没有学明白不要紧，这里我们就用基础代码
# 方法：经典统计迭代、最优子集、Lasso、弹性网络、随机森林

#### 特征工程_不相关的特征 ####
# 无论算法是回归（预测数字）还是分类（预测类别），特征都必须与目标相关。如果一个特征没有表现出相关性，它就是一个主要的消除目标

# 方法一，有点慢
df <- read.csv('trainset.csv') %>% 
  .[,c(2, 4, 6:17, 20:25)]
task <- as_task_regr(df, target = "Distant_Metastasis")
autoplot(task, type = "pairs")
# 响应变量和预测变量的相关性都还可以，可以纳入
# https://zhuanlan.zhihu.com/p/421779689 回归分析预测变量相关性解析
# 当0<| r |≤0.3时，为微弱相关；
# 当0.3<| r |≤0.5时，为低度相关；
# 当0.5<| r |≤0.8时，为显著相关；
# 当0.8<| r |≤1时，为高度相关。
# 要求：预测变量与响应变量之间相关性至少大于0.3，预测变量与预测变量之间相关性至少小于0.8

# 方法二
filter = flt("correlation")
filter$calculate(task)
filter$scores
autoplot(filter)

# 方法三
cor <- cor(df); cor
library(corrplot)
corrplot(cor) # 绘图很慢

#### 特征工程_低方差特征 ####
# 所谓的低方差特征即在不同的响应变量下，预测变量的数值都差不多（低方差）
# 举个栗子：我们想要预测一个人是否患病，我纳入的一个预测变量是人种，而且我的研究是亚洲人是否患病，那么显然都是黄种人，所以这个预测变量就没有意义
# 所以低方差的预测变量可能需要考虑删除

# 方法一
filter = flt("variance")
filter$calculate(task)
filter$scores
as.data.table(filter)

# 方法二
var(df)

#### 特征工程_多重共线性 ####
# 多重共线性的诊断我们主要通过方差膨胀因子，如果大于5则提示有多重共线性
# 当任何两个特征之间存在相关性时，就会出现多重共线性。在机器学习中，期望每个特征都应该独立于其他特征，即它们之间没有共线性

model <- lm(Distant_Metastasis ~ ., data = df)
summary(model)
vif(model) # 查看方差膨胀因子

##### 单因素多因素logistic回归 #####

rm(list = ls())
library(rms)
library(broom)
library(scales)
library(autoReg)

data <- read.csv('trainset.csv') %>% 
  .[,c(2, 4, 6:12, 14:16, 18:19, 22:25)]

{
  data$Sex <- factor(data$Sex,
                     levels = c(1,2),
                     labels = c("Male","Female"))
  data$Race <- factor(data$Race,
                      levels = c(1,2,3),
                      labels = c("White","Black","Other"))
  data$Age <- factor(data$Age,
                     levels = c(1,2,3),
                     labels = c("<1","1-4",">5"))
  data$Household_Income<- factor(data$Household_Income,
                                 levels = c(1,2),
                                 labels = c("<$70,000",">=$70,000"))
  data$Household_Location <- factor(data$Household_Location,
                                    levels = c(1,2),
                                    labels = c("Rural","Urban"))
  data$Tumor_Primary_Site <- factor(data$Tumor_Primary_Site,
                                    levels = c(1,2,3),
                                    labels = c("Adrenal gland","Retroperitoneum","Other"))
  data$Histology <- factor(data$Histology,
                           levels = c(1,2),
                           labels = c("Neuroblastoma","Ganglioneuroblastoma"))
  data$Grade <- factor(data$Grade,
                       levels = c(1,2,3,4),
                       labels = c("I-II","III","IV","Unknown"))
  data$Laterality <- factor(data$Laterality,
                            levels = c(1,2,3,4),
                            labels = c("Bilateral","Unilateral","Not a paired site","Unknown"))
  data$Tumor_Size <- factor(data$Tumor_Size,
                            levels = c(1,2,3),
                            labels = c("<51.5mm",">51.5mm","Unknown"))
  data$First_Malignant_Primary_Indicator <- factor(data$First_Malignant_Primary_Indicator,
                                                   levels = c(0,1),
                                                   labels = c("No","Yes"))
  data$Number_of_Tumors <- factor(data$Number_of_Tumors,
                                  levels = c(1,2),
                                  labels = c("1",">1"))
  data$Surgery_Type <- factor(data$Surgery_Type,
                              levels = c(0,1,2,3),
                              labels = c("No Surgery","Local tumor destruction/excision","Partial surgical removal of primary site","Total surgical removal of primary site"))
  data$Lymph_Nodes_Surgery <- factor(data$Lymph_Nodes_Surgery,
                                     levels = c(0,1,2),
                                     labels = c("No or biopsy only","Regional lymph nodes removed","Unknown"))
  data$Regional_Lymph_Nodes <- factor(data$Regional_Lymph_Nodes,
                                      levels = c(0,1,2,3),
                                      labels = c("No nodes were examined","Negative","Positive","Unknown"))
  data$Chemotherapy <- factor(data$Chemotherapy,
                              levels = c(0,1),
                              labels = c("No","Yes"))
  data$Radiation <- factor(data$Radiation,
                           levels = c(0,1),
                           labels = c("No","Yes"))
  data$Distant_Metastasis <- factor(data$Distant_Metastasis,
                                    levels = c(0,1),
                                    labels = c("No","Yes"))
}

vars = names(data)[c(1:16,18)]; vars

# 批量单因素logistic回归
uni = tibble(vars) %>% 
  mutate(model = map(data[vars], ~ glm(Distant_Metastasis ~ .x, data = data, family = binomial()))) %>% 
  mutate(result = map(model, tidy),
         OR = map(model, ~ exp(coef(.x))),
         OR_ci = map(model, ~ exp(confint(.x)))) %>% 
  dplyr::select(-model) %>% 
  unnest(c(result, OR, OR_ci))

# 生成结果
uni_result = uni %>% 
  mutate(OR_ci %>% as_tibble()) %>% 
  dplyr::select(-OR_ci) %>% 
  rename(LL = V1, UL = V2) %>%  
  mutate(across(term, ~ str_remove(.x, '.x'))) %>% 
  filter(if_all(term, ~ !.x=='(Intercept)')) %>% 
  mutate(`OR(95%CI)` = str_c(round(OR,2), ' (', round(LL,2), '-', round(UL,2), ')')) %>% 
  dplyr::select(vars, term, `OR(95%CI)`, p.value, OR, LL, UL, ) %>% 
  mutate(p.value = pvalue(p.value))

uni_var <- unique(uni_result$vars[uni_result$p.value < 0.05]); uni_var

# 写出结果
write.csv(uni_result,"uni_log_result.csv")

# 多因素logistic回归
# data1
data1 <- read.csv('trainset.csv') %>% 
  .[,c(2, 4, 6:12, 14:16, 18:19, 22:25)]
for (i in names(data1)[c(1:18)]){data1[,i] <- as.factor(data1[,i])}

fit1 = glm(as.formula(paste("Distant_Metastasis ~",
                     paste(uni_var, collapse = "+"))),
          data = data1,
          family = "binomial")

# 提取结果
mul_result1 <- fit1 %>% tidy()
mul_result1$term_sub <- sub("\\d+$", "", mul_result1$term)
mul_var <- unique(mul_result1$term_sub[mul_result1$p.value < 0.05]); mul_var

# 保存多因素logistic有显著性的变量
saveRDS(mul_var, "mul_var.Rdata")

# data
fit = glm(as.formula(paste("Distant_Metastasis ~",
                            paste(uni_var, collapse = "+"))),
           data = data,
           family = "binomial")

# 大致看一下
summary(fit)
autoReg(fit) %>% myft() # 自动作图生成结果

# 提取结果
or_ci <- exp(confint(fit))
p_values <-summary(fit)$coefficients[, 4]
mul_result <- data.frame(
  term = rownames(or_ci)[-1],
  OR = exp(coef(fit))[-1],
  Lower_CI = or_ci[-1, 1],
  Upper_CI = or_ci[-1, 2],
  p.value = p_values[-1])

# 置信区间
mul_result$'OR(95%CI)' <- paste0(round(mul_result$OR,2),"(",
                                 round(mul_result$Lower_CI,2),"-",
                                 round(mul_result$Upper_CI,2),")")

# 创建一个空的数据框，用于存储结果
split_terms <- data.frame(Category = character(), Value = character(), stringsAsFactors = FALSE)

# 遍历 mul_result$term 中的每个元素
for(term in mul_result$term) {
  # 查找与 mul_result1$term_sub[-1] 中任一元素相匹配的部分
  category <- sapply(mul_result1$term_sub[-1], function(x) ifelse(grepl(x, term), x, NA))
  category <- na.omit(category)[1] # 选取第一个匹配项（如果有的话）
  
  # 移除匹配的部分，获取剩余字符串
  value <- gsub(pattern = category, replacement = "", x = term)
  
  # 将结果添加到数据框中
  split_terms <- rbind(split_terms, data.frame(Category = category, Value = value))
}

mul_result <- cbind(split_terms,mul_result)

### 可视化
mul_result$term <- factor(mul_result$term,
                          levels = rev(mul_result$term))
mul_result %>%
  mutate(Change = case_when(
    p.value<0.05 & OR>1 ~ "Risk factor",
    p.value<0.05 & OR<1 ~ "Protective factor",
    p.value>=0.05 ~ "Not sig"
  )) -> mul_result

# 写出结果
write.csv(mul_result,"mul_log_result.csv")

ggplot(mul_result, 
       aes(x = OR, y = term)) +
  geom_errorbarh(aes(xmin = Lower_CI, xmax = Upper_CI, color = Change), 
                 height = 0.2, size = 1) +
  geom_point(alpha = 0.3, aes(size = desc(p.value))) +
  xlim(min(or_ci), max(or_ci)) +
  labs(x = "OR (95%CI)", y = "") +
  geom_vline(xintercept = 1.0, color = "grey", linetype = 2, linewidth = 1) +
  scale_color_manual(values = c("#377eb8","#4daf4a","#e41a1c")) +
  theme_bw() +
  theme(axis.text = element_text(size = 12, color = "black"),
        panel.border = element_rect(linewidth = 1))

mul_result2 <- mul_result[mul_result$Change == "Risk factor" | mul_result$Change == "Protective factor",]

ggplot(mul_result2, 
       aes(x = OR, y = term)) +
  geom_errorbarh(aes(xmin = Lower_CI, xmax = Upper_CI, color = Change), 
                 height = 0.2, size = 1) +
  geom_point(alpha = 0.3, aes(size = desc(p.value))) +
  scale_size_continuous(range = c(2, 10) )+
  xlim(min(or_ci), max(or_ci)) +
  labs(x = "OR (95%CI)", y = "") +
  geom_vline(xintercept = 1.0, color="grey", linetype=2, linewidth=1) +
  scale_color_manual(values = c("#4daf4a","#e41a1c")) +
  theme_bw() +
  theme(axis.text = element_text(size = 12, color = "black"),
        panel.border = element_rect(linewidth = 1))

