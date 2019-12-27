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
data = read.csv("train.csv")
test_file = read.csv("test.csv")
#################################################################################
"""Exploring dataset"""
summary(data)
glimpse(data)
glimpse(test_file)
summary(test_file)
#there is no NA in the dataset train and test
sum(is.na(data))
#Dropping ID_code variable and changing index to ID_code
data = data[2:202]
data$target = factor(data$target, levels = c(0, 1))
test_id = test_file[,1]
test_file= test_file[2:201]
#Finding correlation
corr = cor.test(data$var_0, data$target,  method = "pearson", use = "complete.obs")
data[1:200,2:3]%>%
  ggplot(aes(var_0,var_1))+
  geom_point()

"""Shapiro-Wilk test can be performed as follow:
       Null hypothesis: the data are normally distributed
       Alternative hypothesis: the data are not normally distributed"""
#Train data
shapiro.test(data$var_115[1:5000]) #p-value = 1.254e-08
shapiro.test(data$var_5[1:5000])#p-value < 2.2e-16
shapiro.test(data$var_115[15001:20000])#p-value = 1.524e-09
#Test data
shapiro.test(test$var_115[15001:20000])#p-value = 1.372e-05
#hence we reject the null hypothesis and say the result is statistically significant

#Q-Q plot for checking normal distribution
#Train
data%>%
  ggplot(aes(sample = var_3))+
  geom_qq()+ geom_qq_line()+
  labs(title = "Q-Q Plot")
data%>%
  ggplot(aes(sample = var_199))+
  geom_qq()+ geom_qq_line()+
  labs(title = "Q-Q Plot")

#Test
test_file%>%
  ggplot(aes(sample = var_199))+
  geom_qq()+ geom_qq_line()+
  labs(title = "Q-Q Plot")

#from the correlation heatmap we can see that there is no high correlation between
#any two variables.

#################################################################################
"""DATA Visualization"""
#Grouping by target variable to get the count of 0 & 1
data%>%
  group_by(target)%>%
  count()%>%
  ggplot(aes(target,n,fill = target))+
  geom_col()+
  labs(y = "Count", title = "Target variable caount")
#179902 of 0 and 20098 of 1

#Visualization of Variance and Standard deviation of all columns
#train data
sd_df = data.frame('sd_train' = c(sapply(data[2:201], sd, na.rm = TRUE)))
sd_df$variance_train = c(sapply(data[2:201], var, na.rm = TRUE))
sd_df$sd_test = c(sapply(test_file[1:200], var, na.rm = TRUE))
sd_df$variance_test = c(sapply(test_file[1:200], var, na.rm = TRUE))

#for Standard deviation of Train data
sd_df%>%
  ggplot(aes(y = sd_train, x = row.names(sd_df)))+
  geom_col()+
  coord_flip()+
  labs(y = "Standard Deviation",x="Variables",
       title = "Horizontal display of standard deviation of variables of Train")

#for Variance of Train data
sd_df%>%
  ggplot(aes(y = variance_train, x = row.names(sd_df)))+
  geom_col()+
  coord_flip()+
  labs(y = "Variance",x="Variables",
       title = "Horizontal display of variance of variables of Train")

#value of standard deviation vs count
sd_df%>%
  ggplot(aes(sd_train))+
  geom_histogram(bins = 100)+
  labs(title = "value of standard deviation vs count Train")

#value of Variance vs count
sd_df%>%
  ggplot(aes(variance_train))+
  geom_histogram(bins = 100)+
  labs(title = "value of Variance vs count of Train")

#Test Data
#for Standard deviation of Test data
sd_df%>%
  ggplot(aes(y = sd_test, x = row.names(sd_df)))+
  geom_col()+
  coord_flip()+
  labs(y = "Standard Deviation",x="Variables",
       title = "Horizontal display of standard deviation of variables of Test")

#for Variance of Test data
sd_df%>%
  ggplot(aes(y = variance_test, x = row.names(sd_df)))+
  geom_col()+
  coord_flip()+
  labs(y = "Variance",x="Variables",
       title = "Horizontal display of variance of variables of Test")

#value of standard deviation vs count
sd_df%>%
  ggplot(aes(sd_test))+
  geom_histogram(bins = 100)+
  labs(title = "value of standard deviation vs count Test")

#value of Variance vs count
sd_df%>%
  ggplot(aes(variance_test))+
  geom_histogram(bins = 100)+
  labs(title = "value of Variance vs count of Test")
#standard deviation and variance of almost all variables is very close to each other
#which means that the data is not diversified much. This may include is=n difficulties
#to predict the target variable as values are very close to each other

#Comparing STD of both test and train data
comp01 = sd_df%>%
  ggplot(aes(sd_train))+
  geom_histogram(bins = 200)+
  labs(title = "Comparing STD of both test and train data")
comp02 = sd_df%>%
  ggplot(aes(sd_test))+
  geom_histogram(bins = 200)+
  labs(title = "Comparing STD of both test and train data")
multi_plot01 = marrangeGrob(list(comp01,comp02),nrow = 2,ncol=1)

#Visualisation of few data points on a scatter plot of train vs test
plot_dist = function(df1,df2,features,nrows,ncols){
  fig_list = list()
  j=1
  for (feature in features){
    print(feature)
    fig=ggplot()+
      geom_density(data = df1,aes_string(x=feature),color="purple")+
      geom_density(data = df2,aes_string(x=feature),color = "orange")
    fig_list[j] = list(fig)
    j = j+1  
  }
  multi_plot02 = marrangeGrob(grobs =  fig_list,nrow = nrows,ncol = ncols)
  return(multi_plot02)
}
features = colnames(data1)
multi_plot02=plot_dist(data,test_file,colnames(data)[2:30],5,4)
#from this we got to that train data and test data are on same range as per their values

#Comparing train and test mean values
data_mean= list()
j=1
for (i in colnames(data[2:201])){
  data_mean[j] = mean(data[[i]])
  j=j+1
}
data_mean  = unlist(data_mean, use.names = FALSE)
test_mean = list()
j=1
for (i in colnames(test_file[1:200])){
  test_mean[j] = mean(test_file[[i]])
  j=j+1
}
test_mean  = unlist(test_mean, use.names = FALSE)
ggplot()+
  geom_histogram(aes(data_mean,fill = "purple"),bins=200)+
  geom_histogram(aes(test_mean,fill="orange"),bins=200)+
  labs(title = "Mean values of Test and Train data")

#plotting graph of variables w.r.t target for comparison
fig_list = list()
j = 1
for (i in colnames(data[2:201])){
    fig=ggplot(data=data,aes_string(x = i,color = "target"))+
      geom_density()+
      labs(title = "Coparision of variables w.r.t Target variable")
      labs(x=i)
    fig_list[j] = list(fig)
  j = j+1
  print(j)
}
multi_plot02=marrangeGrob(fig_list,nrow = 4,ncol = 5)

#Ploting graph by row, to observe any pattern in variables combining to give 1 and 0
t0 = data%>%
  filter(target == 0)
t1 = data%>%
  filter(target == 1)

plot_dist1 = function(df1,df2,nrows,ncols){
  fig_list = list()
  j=1
  for (count in (1:dim(df1)[1])){
    print(count)
    fig=ggplot()+
      geom_density(data = melt(df1[count,]),aes(x=value),color="purple")+
      geom_density(data = melt(df2[count,]),aes(x=value),color = "orange")
    fig_list[j] = list(fig)
    j = j+1  
  }
  multi_plot = marrangeGrob(grobs =  fig_list,nrow = nrows,ncol = ncols)
  return(multi_plot)
}
multi_plot02=plot_dist1(t0[2:10,],t1[2:10,],5,4)

#################################################################################
"""Model Building"""
########################----Model set 01----#################################
"""All models in this set are based on normal unteweked dataset"""
#Splitting data into train and test 
set.seed(1234)
train_index = createDataPartition(data$target,p=0.8,list = FALSE)
train = data[train_index,]
test = data[-train_index,]

"""Logistic Regression"""
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
