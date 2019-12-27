rm(list = ls())
#setwd("D:/MY PROGRAMMING DATA/edwisor/Project01")

#################################################################################
#Libraries to import
library(data.table) #data wrangling
library(tibble) #data wrangling
library(ggplot2) #visualization
library(dplyr)
library(gridExtra)#for multiplot
library(caret)#data manipulation
library(e1071) #Naive Bayes
library(rpart) #Decission Tree
library('randomForest') #random forest
library(class) #KNN
library(pROC) #ROC
library(DMwR) #SMOTE
library(ROSE)# Oversampling


#################################################################################
"""Importing dataset"""
print("Importing dataset")
data = read.csv("train.csv")
test_file = read.csv("test.csv")
#################################################################################
"""Exploring dataset"""

print(summary(data))
print(glimpse(data))
print(glimpse(test_file))
print(summary(test_file))
#there is no NA in the dataset train and test
print(sum(is.na(data)))
#Dropping ID_code variable and changing index to ID_code
data = data[2:202]
data$target = factor(data$target, levels = c(0, 1))
test_id = test_file[,1]
test_file= test_file[2:201]
#Finding correlation
corr = cor.test(data$var_0, data$target,  method = "pearson", use = "complete.obs")
corr
#from the correlation heatmap we can see that there is no high correlation between
#any two variables.

print("Shapiro-Wilk test can be performed as follow:
Null hypothesis: the data are normally distributed
Alternative hypothesis: the data are not normally distributed")

#Train data
print(shapiro.test(data$var_115[1:5000])) #p-value = 1.254e-08
print(shapiro.test(data$var_5[1:5000]))#p-value < 2.2e-16
print(shapiro.test(data$var_115[15001:20000]))#p-value = 1.524e-09
#Test data
print(shapiro.test(test_file$var_115[15001:20000]))#p-value = 1.372e-05
#hence we reject the null hypothesis and say the result is statistically significant


#################################################################################
"""Model Building"""
########################----Model set 01----#################################
print("########################----Model set 01----#################################")
"""All models in this set are based on normal unteweked dataset"""
#Splitting data into train and test 
set.seed(1234)
train_index = createDataPartition(data$target,p=0.8,list = FALSE)
train = data[train_index,]
test = data[-train_index,]

"""Logistic Regression"""
print("Logistic Regression")
classifier = glm(target ~., data=train , family="binomial")
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:201], type = "response")
#changing probability to 0 and 1
y_pred = ifelse(y_pred > 0.5,1,0)
#Confusion matrics
cm = table(test$target, y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)

"""NaiveBayes"""
print("NaiveBayes")
classifier = naiveBayes(target ~. , data = train)
#Prediction
y_pred = predict(classifier , newdata=test[,2:201])
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Decision Tree classifier"""
print("Decision Tree classifier")
classifier = rpart(formula = target ~. , data = train)
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:201], type = 'class')
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Random FOrest Classification"""
print("Random FOrest Classification")
set.seed(123)
classifier = randomForest(x = train[,2:201],y = train$target ,ntree = 10)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test[,2:201])
# Making the Confusion Matrix
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""KNN"""
print("KNN")
train[,2:201] = scale(train[,2:201])
test[,2:201] = sclae(test[,2:201])

classifier = knn(train[,2:201], test[,2:201], train$target, k=3)
#Confusion matrics
cm = table(test$Purchased , classifier)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

########################----Model set 02----#################################
print("########################----Model set 02----#################################")
"""In this odel we are increasing the minority class by SMOTE"""
#Splitting data into train and test 
set.seed(1234)
train_index = createDataPartition(data$target,p=0.8,list = FALSE)
train = data[train_index,]
test = data[-train_index,]
#SMOTE
train%>%
  group_by(target)%>%
  count()
train = SMOTE(target ~ ., train, perc.over = 100)
train%>%
  group_by(target)%>%
  count()


"""Logistic Regression"""
print("Logistic Regression")
classifier = glm(target ~., data=train , family="binomial")
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:201], type = "response")
#changing probability to 0 and 1
y_pred = ifelse(y_pred > 0.5,1,0)
#Confusion matrics
cm = table(test$target, y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""NaiveBayes"""
print("NaiveBayes")
classifier = naiveBayes(target ~. , data = train)
#Prediction
y_pred = predict(classifier , newdata=test[,2:201])
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Decision Tree classifier"""
print("Decision Tree classifier")

classifier = rpart(formula = target ~. , data = train)
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:201], type = 'class')
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Random FOrest Classification"""
print("Random FOrest Classification")
set.seed(123)
classifier = randomForest(x = train[,2:201],y = train$target ,ntree = 10)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test[,2:201])
# Making the Confusion Matrix
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""KNN"""
print("KNN")
train[,2:201] = scale(train[,2:201])
test[,2:201] = sclae(test[,2:201])

classifier = knn(train[,2:201], test[,2:201], train$target, k=3)
#Confusion matrics
cm = table(test$Purchased , classifier)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

########################----Model set 03----#################################
print("########################----Model set 03----#################################")
"""Upsampling the dataset"""
#Splitting data into train and test 
set.seed(1234)
train_index = createDataPartition(data$target,p=0.8,list = FALSE)
train = data[train_index,]
test = data[-train_index,]

train%>%
  group_by(target)%>%
  count()

#Over sampling
train = ovun.sample(target ~ ., data = train, method = "over",N = 287844)$data
train%>%
  group_by(target)%>%
  count()

"""Logistic Regression"""
print("Logistic Regression")
classifier = glm(target ~., data=train , family="binomial")
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:201], type = "response")
#changing probability to 0 and 1
y_pred = ifelse(y_pred > 0.5,1,0)
#Confusion matrics
cm = table(test$target, y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""NaiveBayes"""
print("NaiveBayes")
classifier = naiveBayes(target ~. , data = train)
#Prediction
y_pred = predict(classifier , newdata=test[,2:201])
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Decision Tree classifier"""
print("Decision Tree classifier")
classifier = rpart(formula = target ~. , data = train)
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:201], type = 'class')
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Random FOrest Classification"""
print("Random FOrest Classification")
set.seed(123)
classifier = randomForest(x = train[,2:201],y = train$target ,ntree = 10)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test[,2:201])
# Making the Confusion Matrix
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""As there is no such improvement in KNN so we will not consider it from now"""

########################----Model set 04----#################################
print("########################----Model set 04----#################################")
"""Dropping few variables from the dataset as those variables conatins same value for 
0 and 1. Data will be upsampled as the best result was seen in upsampled dataset"""
set.seed(1234)
train_index = createDataPartition(data$target,p=0.8,list = FALSE)
train = data[train_index,]
test = data[-train_index,]

#drop variable
train = select(train,-c('var_17','var_38','var_46','var_59','var_61','var_65',
                        'var_73','var_79','var_96','var_100','var_124','var_185'))
test = select(test,-c('var_17','var_38','var_46','var_59','var_61','var_65',
                      'var_73','var_79','var_96','var_100','var_124','var_185'))
#checking count
train%>%
  group_by(target)%>%
  count()

#Over sampling
train = ovun.sample(target ~ ., data = train, method = "over",N = 287844)$data
train%>%
  group_by(target)%>%
  count()

"""Logistic Regression"""
print("Logistic Regression")
classifier = glm(target ~., data=train , family="binomial")
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:189], type = "response")
#changing probability to 0 and 1
y_pred = ifelse(y_pred > 0.5,1,0)
#Confusion matrics
cm = table(test$target, y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""NaiveBayes"""
print("NaiveBayes")
classifier = naiveBayes(target ~. , data = train)
#Prediction
y_pred = predict(classifier , newdata=test[,2:189])
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Decision Tree classifier"""
print("Decision Tree classifier")

classifier = rpart(formula = target ~. , data = train)
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:189], type = 'class')
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Random FOrest Classification"""
print("Random FOrest Classification")
set.seed(123)
classifier = randomForest(x = train[,2:189],y = train$target ,ntree = 10)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test[,2:189])
# Making the Confusion Matrix
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

########################----Model set 05----#################################
print("########################----Model set 05----#################################")
"""Dropping few variables from the dataset as those variables conatins same value for 
0 and 1. Data will be SMOTE as the best result was seen in upsampled dataset"""

train = select(train,-c('var_17','var_38','var_46','var_59','var_61','var_65',
                        'var_73','var_79','var_96','var_100','var_124','var_185'))
test = select(test,-c('var_17','var_38','var_46','var_59','var_61','var_65',
                      'var_73','var_79','var_96','var_100','var_124','var_185'))

#SMOTE
train%>%
  group_by(target)%>%
  count()
train = SMOTE(target ~ ., train, perc.over = 100)
train%>%
  group_by(target)%>%
  count()

"""Logistic Regression"""
print("Logistic Regression")
classifier = glm(target ~., data=train , family="binomial")
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:189], type = "response")
#changing probability to 0 and 1
y_pred = ifelse(y_pred > 0.5,1,0)
#Confusion matrics
cm = table(test$target, y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""NaiveBayes"""
print("NaiveBayes")
classifier = naiveBayes(target ~. , data = train)
#Prediction
y_pred = predict(classifier , newdata=test[,2:189])
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Decision Tree classifier"""
print("Decision Tree classifier")
classifier = rpart(formula = target ~. , data = train)
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:189], type = 'class')
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""Random FOrest Classification"""
print("Random FOrest Classification")
set.seed(123)
classifier = randomForest(x = train[,2:189],y = train$target ,ntree = 10)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test[,2:189])
# Making the Confusion Matrix
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

########################----Model set 06----#################################
print("########################----Model set 07----#################################")
"""Trying some feature enginering with normalization.
We will include:
Square of each row of each column
value - maens of that columns
cube of each row of each column"""

set.seed(1234)
train_index = createDataPartition(data$target,p=0.8,list = FALSE)
train = data[train_index,]
test = data[-train_index,]

features = colnames(train[2:201])
for(feature in features){
  train[paste(feature,"_sq",sep = '')] = train[feature]^2
  train[paste(feature,"_cu",sep = '')] = train[feature]^3
}
for(feature in features){
  test[paste(feature,"_sq",sep = '')] = testfeature]^2
test[paste(feature,"_cu",sep = '')] = test[feature]^3
}

train[,2:601] = scale(train[,2:601])
test[,2:601] = sclae(test[,2:601])

"""Logistic Regression"""
print("Logistic Regression"
classifier = glm(target ~., data=train , family="binomial")
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:601], type = "response")
#changing probability to 0 and 1
y_pred = ifelse(y_pred > 0.5,1,0)
#Confusion matrics
cm = table(test$target, y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""NaiveBayes"""
print("NaiveBayes")
classifier = naiveBayes(target ~. , data = train)
#Prediction
y_pred = predict(classifier , newdata=test[,2:601])
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

########################----Model set 07----#################################
print("########################----Model set 01----#################################")
"""Trying some feature enginering with normalization.
We will include:
Square of each row of each column
rank of each variable values
cube of each row of each column"""

set.seed(1234)
train_index = createDataPartition(data$target,p=0.8,list = FALSE)
train = data[train_index,]
test = data[-train_index,]

features = colnames(train[2:201])
for(feature in features){
  train[paste(feature,"_sq",sep = '')] = train[feature]^2
  train[paste(feature,"_cu",sep = '')] = train[feature]^3
  train[paste(feature,"_ra",sep = '')] = rank(train[feature])
}
for(feature in features){
  test[paste(feature,"_sq",sep = '')] = testfeature]^2
test[paste(feature,"_cu",sep = '')] = test[feature]^3
test[paste(feature,"_ra",sep = '')] = rank(test[feature])
}

train%>%
  group_by(target)%>%
  count()

#Over sampling
train = ovun.sample(target ~ ., data = train, method = "over",N = 287844)$data
train%>%
  group_by(target)%>%
  count()


train[,2:801] = scale(train[,2:801])
test[,2:801] = sclae(test[,2:801])


"""Logistic Regression"""
print("Logistic Regression")
classifier = glm(target ~., data=train , family="binomial")
summary(classifier)
#Prediction
y_pred = predict(classifier, newdata = test[,2:801], type = "response")
#changing probability to 0 and 1
y_pred = ifelse(y_pred > 0.5,1,0)
#Confusion matrics
cm = table(test$target, y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

"""NaiveBayes"""
print("NaiveBayes")
classifier = naiveBayes(target ~. , data = train)
#Prediction
y_pred = predict(classifier , newdata=test[,2:801])
#Confusion matrics
cm = table(test[,1], y_pred)
#ROC
auc = roc(test$target, y_pred)
print(auc)
plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)

#Creating Submission file
#set.seed(1234)
#train_index = createDataPartition(data$target,p=0.8,list = FALSE)
#train = data[train_index,]
#test = data[-train_index,]
#train = ovun.sample(target ~ ., data = train, method = "over",N = 287844)$data
#classifier = naiveBayes(target ~. , data = train)
#Prediction
#y_pred = predict(classifier , newdata=test_file)
#result = data.frame('ID_code' = test_id,'target' = y_pred)
#write.csv(result, file = "r_result.csv",row.names=FALSE)

#########################################################################################

