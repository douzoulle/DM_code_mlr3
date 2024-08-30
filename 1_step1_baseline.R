
library(foreign)
library(survival)
library(rms)
library(nomogramFormula)
library(rsconnect)
library(survminer)
library(forestplot)  
library(compareGroups)  
library(data.table)
library(readxl)

rm(list = ls())

# 读取数据
table <- read_excel("nb_fenxi.xlsx") %>% 
  as.data.frame() %>% 
  .[, -1]
table$Age_Continuous <- as.numeric(sub("^(\\d{2}).*", "\\1", table$Age_Continuous))

# 查看na
sum(is.na(table)) # na个数
which(is.na(table), arr.ind = TRUE) # na位置

# 寻找tumor size最佳截断值
library(pROC)
proc <- subset(table, Tumor_Size != "unknown")
proc$Tumor_Size <- as.numeric(proc$Tumor_Size)

# 基于约登指数最大的点选择cutoff值
proc1 <- roc(proc$Distant_Metastasis, proc$Tumor_Size, ci = T);proc1 # 逗号前为结局，后为连续变量
plot(proc1,
     print.auc = TRUE, # 显示AUC
     print.thres = TRUE, # 显示最佳cutoff
     main = "ROC for Tumor Size",
     col = "#008600") 
# 最佳截断值为51.5mm

# tumor size连续变量转化为分类变量
table(table$Tumor_Size)
table$Tumor_Size <- ifelse(table$Tumor_Size %in% "unknown", 3,
                          ifelse(table$Tumor_Size >= 52, 2, 1))
table(table$Tumor_Size)

# 7:3随机分为训练集验证集
ind <- sample(2, nrow(table), replace = TRUE, prob = c(0.7,0.3))
set.seed(123)
trainset <- table[ind == 1,]
testset <- table[ind == 2,]
write.csv(trainset, "trainset.csv")
write.csv(testset, "testset.csv")

trainset$Group <- "1"
testset$Group <- "2"

# 合并训练集验证集
data <- rbind(trainset, testset)
str(data)
head(data)
names(data)

# 转换为因子变量
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
  data$Year_of_Diagnosis <- factor(data$Year_of_Diagnosis,
                                   levels = c(1,2),
                                   labels = c("2000-2010","2011-2020"))
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
  data$Summary_Stage <- factor(data$Summary_Stage,
                               levels = c(1,2,3,4),
                               labels = c("Localized","Regional","Distant","Unknown/Unstaged"))
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
  data$Surgery_Other_Sites <- factor(data$Surgery_Other_Sites,
                                     levels = c(0,1),
                                     labels = c("No","Yes"))
  data$Radiation <- factor(data$Radiation,
                           levels = c(0,1),
                           labels = c("None/Unknown","Yes"))
  data$Chemotherapy<- factor(data$Chemotherapy,
                             levels = c(0,1),
                             labels = c("None/Unknown","Yes"))
  data$Radiotherapy_Sequence <- factor(data$Radiotherapy_Sequence,
                                       levels = c(0,1,2,3,4),
                                       labels = c("None/Unknown","radiation before surgery","radiation after surgery","before and after","Unknown"))
  data$Chemotherapy_Sequence <- factor(data$Chemotherapy_Sequence,
                                       levels = c(0,1,2,3),
                                       labels = c("None/Unknown","chemotherapy before surgery","chemotherapy after surgery","before and after"))
  data$Bone_Metastasis <- factor(data$Bone_Metastasis,
                                 levels = c(0,1,2),
                                 labels = c("No","Yes","Unknown"))
  data$Brain_Metastasis <- factor(data$Brain_Metastasis,
                                  levels = c(0,1,2),
                                  labels = c("No","Yes","Unknown"))
  data$Liver_Metastasis <- factor(data$Liver_Metastasis,
                                  levels = c(0,1,2),
                                  labels = c("No","Yes","Unknown"))
  data$Lung_Metastasis <- factor(data$Lung_Metastasis,
                                 levels = c(0,1,2),
                                 labels = c("No","Yes","Unknown"))
  data$Distant_LN_Metastasis <- factor(data$Distant_LN_Metastasis,
                                       levels = c(0,1,2),
                                       labels = c("No","Yes","Unknown"))
  data$Other_Distant_Metastasis <- factor(data$Other_Distant_Metastasis,
                                          levels = c(0,1,2),
                                          labels = c("No","Yes","Unknown"))
  data$Distant_Metastasis <- factor(data$Distant_Metastasis,
                                    levels = c(0,1),
                                    labels = c("No","Yes"))
  data$Vital_Status <- factor(data$Vital_Status,
                              levels = c(0,1),
                              labels = c("Alive","Dead"))
  data$Cancer_Specific_Death <- factor(data$Cancer_Specific_Death,
                                       levels = c(0,1),
                                       labels = c("Not cancer specific death","Dead due to bladder cancer"))
  data$Group <- factor(data$Group,
                       levels = c(1,2),
                       labels = c("Training Cohort","Validation Cohort"))
}
data$Survival_Months<- as.numeric(data$Survival_Months)
str(data)

# 开始画基线图
names(data)
descrTable(Group ~ Sex + Age + Age_Continuous + Race + Year_of_Diagnosis + Household_Location + Household_Income +
             Tumor_Primary_Site + Laterality + Histology + Grade + Summary_Stage + Tumor_Size + Surgery_Type + 
             Lymph_Nodes_Surgery + Regional_Lymph_Nodes + Surgery_Other_Sites + Chemotherapy + Chemotherapy_Sequence +
             Radiation + Radiotherapy_Sequence + First_Malignant_Primary_Indicator + Number_of_Tumors + Distant_Metastasis + 
             Bone_Metastasis + Brain_Metastasis + Liver_Metastasis + Lung_Metastasis + Distant_LN_Metastasis + 
             Other_Distant_Metastasis + Survival_Months + Vital_Status + Cancer_Specific_Death,
           data = data)
restab <- descrTable(Group ~ Sex + Age + Age_Continuous + Race + Year_of_Diagnosis + Household_Location + Household_Income +
                       Tumor_Primary_Site + Laterality + Histology + Grade + Summary_Stage + Tumor_Size + Surgery_Type + 
                       Lymph_Nodes_Surgery + Regional_Lymph_Nodes + Surgery_Other_Sites + Chemotherapy + Chemotherapy_Sequence +
                       Radiation + Radiotherapy_Sequence + First_Malignant_Primary_Indicator + Number_of_Tumors + Distant_Metastasis + 
                       Bone_Metastasis + Brain_Metastasis + Liver_Metastasis + Lung_Metastasis + Distant_LN_Metastasis + 
                       Other_Distant_Metastasis + Survival_Months + Vital_Status + Cancer_Specific_Death,
                    data = data,
                    show.all = TRUE)
export2xls(restab, file = '基线表_nb.xlsx')


##### ex #####
rm(list = ls())
data <- read.csv('ex.csv') %>% .[,-1]

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
  data$Year_of_Diagnosis <- factor(data$Year_of_Diagnosis,
                                   levels = c(1,2),
                                   labels = c("2000-2010","2011-2020"))
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
  data$Summary_Stage <- factor(data$Summary_Stage,
                               levels = c(1,2,3,4),
                               labels = c("Localized","Regional","Distant","Unknown/Unstaged"))
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
  data$Surgery_Other_Sites <- factor(data$Surgery_Other_Sites,
                                     levels = c(0,1),
                                     labels = c("No","Yes"))
  data$Radiation <- factor(data$Radiation,
                           levels = c(0,1),
                           labels = c("None/Unknown","Yes"))
  data$Chemotherapy<- factor(data$Chemotherapy,
                             levels = c(0,1),
                             labels = c("None/Unknown","Yes"))
  data$Radiotherapy_Sequence <- factor(data$Radiotherapy_Sequence,
                                       levels = c(0,1,2,3,4),
                                       labels = c("None/Unknown","radiation before surgery","radiation after surgery","before and after","Unknown"))
  data$Chemotherapy_Sequence <- factor(data$Chemotherapy_Sequence,
                                       levels = c(0,1,2,3),
                                       labels = c("None/Unknown","chemotherapy before surgery","chemotherapy after surgery","before and after"))
  data$Bone_Metastasis <- factor(data$Bone_Metastasis,
                                 levels = c(0,1,2),
                                 labels = c("No","Yes","Unknown"))
  data$Brain_Metastasis <- factor(data$Brain_Metastasis,
                                  levels = c(0,1,2),
                                  labels = c("No","Yes","Unknown"))
  data$Liver_Metastasis <- factor(data$Liver_Metastasis,
                                  levels = c(0,1,2),
                                  labels = c("No","Yes","Unknown"))
  data$Lung_Metastasis <- factor(data$Lung_Metastasis,
                                 levels = c(0,1,2),
                                 labels = c("No","Yes","Unknown"))
  data$Distant_LN_Metastasis <- factor(data$Distant_LN_Metastasis,
                                       levels = c(0,1,2),
                                       labels = c("No","Yes","Unknown"))
  data$Other_Distant_Metastasis <- factor(data$Other_Distant_Metastasis,
                                          levels = c(0,1,2),
                                          labels = c("No","Yes","Unknown"))
  data$Distant_Metastasis <- factor(data$Distant_Metastasis,
                                    levels = c(0,1),
                                    labels = c("No","Yes"))
  data$Vital_Status <- factor(data$Vital_Status,
                              levels = c(0,1),
                              labels = c("Alive","Dead"))
  data$Cancer_Specific_Death <- factor(data$Cancer_Specific_Death,
                                       levels = c(0,1),
                                       labels = c("Not cancer specific death","Dead due to bladder cancer"))
}
data$Survival_Months<- as.numeric(data$Survival_Months)
str(data)

descrTable(Distant_Metastasis ~ Sex + Age + Age_Continuous + Race + Year_of_Diagnosis + Household_Location + Household_Income +
             Tumor_Primary_Site + Laterality + Histology + Grade + Summary_Stage + Tumor_Size + Surgery_Type + 
             Lymph_Nodes_Surgery + Regional_Lymph_Nodes + Surgery_Other_Sites + Chemotherapy + Chemotherapy_Sequence +
             Radiation + Radiotherapy_Sequence + First_Malignant_Primary_Indicator + Number_of_Tumors + 
             Bone_Metastasis + Brain_Metastasis + Liver_Metastasis + Lung_Metastasis + Distant_LN_Metastasis + 
             Other_Distant_Metastasis + Survival_Months + Vital_Status + Cancer_Specific_Death,
           data = data)
restab <- descrTable(Distant_Metastasis ~ Sex + Age + Age_Continuous + Race + Year_of_Diagnosis + Household_Location + Household_Income +
                       Tumor_Primary_Site + Laterality + Histology + Grade + Summary_Stage + Tumor_Size + Surgery_Type + 
                       Lymph_Nodes_Surgery + Regional_Lymph_Nodes + Surgery_Other_Sites + Chemotherapy + Chemotherapy_Sequence +
                       Radiation + Radiotherapy_Sequence + First_Malignant_Primary_Indicator + Number_of_Tumors + 
                       Bone_Metastasis + Brain_Metastasis + Liver_Metastasis + Lung_Metastasis + Distant_LN_Metastasis + 
                       Other_Distant_Metastasis + Survival_Months + Vital_Status + Cancer_Specific_Death,
                     data = data,
                     show.all = TRUE)
export2xls(restab, file = '基线表_nb_ex.xlsx')

