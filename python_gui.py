#!/usr/bin/env python3

"""The dataset both train and test countains all numeric values. There is 
no description of the varibale label. There is no NA value in the dataset 
both test and train. Exploring the train data we found that about 89% of 
of the data countains 0(zero) as the target variable and only 11% countains 
1. So we nned to sample out data or else the model will be baised towards 0"""
#Delete all the variables which contain same data for 0 and 1 and then run models
#plot a sns.distplot for few of the rows to find any difference b/w rows yeilding to
#1 amd rows yeilding to 0

#################################################################################
import os
import pandas as pd
import numpy as np
from numpy import cov
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score,roc_curve
from imblearn.over_sampling import SMOTE #for oversampling
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata,norm, shapiro
#################################################################################

"""Importing dataset"""
#os.chdir('D:\MY PROGRAMMING DATA\edwisor\Project01')
data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#################################################################################
"""Exploring dataset"""

head = data.head()
head = test.head()
del head #created for checking head and found that all are numerical values
data.describe()
na = test.isna().sum() #there is no NA in the dataset train and test
del na
#Dropping ID_code variable and changing index to ID_code
data.index = data.ID_code
data = data.drop('ID_code',axis=1)
testID = test['ID_code']
test = test.drop('ID_code',axis=1)
#Finding correlation
corr = data.corr()
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(corr,cmap=sns.diverging_palette(20, 220, n=200))
ax = plt.title("Correlation Plot")
#from the correlation heatmap we can see that there is no high correlation between
#any two variables.
#normal distribution of data/hyporthesis test
"""Shapiro-Wilk test can be performed as follow:
       Null hypothesis: the data are normally distributed
       Alternative hypothesis: the data are not normally distributed"""
shapiro(data['var_0'][0:5000])#p-value 4.205e-13
shapiro(data['var_115'][0:5000])#p-value 1.258e-08


#################################################################################

"""DATA Visualization"""
#Grouping by target variable to get the count of 0 & 1
data.var_0.groupby(by=data.target).count() #179902 of 0 and 20098 of 1
sns.countplot(x=data.target)
sns.title("Target variable count")


#Visualization of Variance and Standard deviation of all columns
#train data
#horizontal display of var() of variables
plt.subplots(figsize=(18,22))
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.08)
plt.barh(y=data.columns,width=data.var(),orientation='horizontal',height=1)
plt.title("Horizontal display of variance of variables of Train") 
plt.xlabel("Variance")
plt.autoscale(tight=True)

#value of var() vs count
plt.subplots(figsize=(18,22))
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.09)
plt.hist(data.var(),bins=100)
plt.title("Variance vs Count Train")
plt.xlabel("Variance")
plt.ylabel("Count")
plt.autoscale(tight=True)

#horizontal display of std() of variables
plt.subplots(figsize=(18,22))
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.08)
plt.barh(y=data.columns,width=data.std(),orientation='horizontal',height=1)
plt.title("Horizontal Display of Standard Deviation of variables Train")
plt.xlabel("Standard Deviation")
plt.autoscale(tight=True)

#Standard Deviation vs Count Train
plt.subplots(figsize=(18,22))
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.08)
plt.hist(data.std(),bins=200)#value of std() vs count
plt.title("Standard Deviation vs Count Train")
plt.xlabel("Standard Deviation")
plt.ylabel("Count")
plt.xticks(ticks=range(22))
plt.autoscale(tight=True)

#test data
#horizontal display of var() of variables
plt.subplots(figsize=(18,22))
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.08)
plt.barh(y=test.columns,width=test.var(),orientation='horizontal')
plt.title("Horizontal display of variance of variables of Test")
plt.xlabel("Variance")
plt.autoscale(tight=True)

#value of var() vs count
plt.subplots(figsize=(18,22))
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.08) 
plt.hist(test.var(),bins=100)
plt.title("Variance vs Count Test")
plt.xlabel("Variance")
plt.ylabel("Count")
plt.autoscale(tight=True)

#horizontal display of std() of variables
plt.subplots(figsize=(18,22))
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.08) 
plt.barh(y=test.columns,width=test.std(),orientation='horizontal',height=1)#horizontal display of std() of variables
plt.title("horizontal display of std() of variables Test")
plt.xlabel("Standard Deviation")
plt.autoscale(tight=True)

#value of std() vs count
plt.subplots(figsize=(18,22))
plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.08) 
plt.hist(test.std(),bins=200)#value of std() vs count
plt.title("Standard Deviation vs Count Test")
plt.xlabel("Standard Deviation")
plt.ylabel("Count")
plt.xticks(ticks=range(22))
plt.autoscale(tight=True)
#standard deviation and variance of almost all variables is very close to each other
#which means that the data is not diversified much. This may include is=n difficulties
#to predict the target variable as values are very close to each other

#Comparing STD of both test and train data
plt.subplot(2,1,1)
plt.hist(data.std(),bins=200)
plt.title("Train Standard Devation Histogram")

plt.subplot(2,1,2)
plt.hist(test.std(),bins=200)#value of std() vs count
plt.title("test standard Deviation Histogram")
#Both the test data nad train data has almost same pattern and hence we can assume that
#both the datasets are in the same range.

#Difference in Mean and standard deviation of two target classes
plt.hist(data[data['target'] ==0].mean() - data[data['target'] ==1].mean())
plt.title('Histogram of Sample Mean Differences between Two Classes')

plt.hist(data[data['target']  == 0].std() - data[data['target'] == 1].std())
plt.title('Histogram of Sample Variance Differences between Two Classes')

#Comparing train and test mean values
fig=plt.subplots(figsize=(10,20))
#axx=plt.subplot(111)
sns.distplot(data.drop('target',axis=1).mean(),color='red',label = "Train",bins=120)
sns.distplot(test.mean(),color='blue',label = "test",bins=120)
plt.xlabel("mean")
plt.ylabel("count")
plt.title("Comparison of mean")
plt.legend()
plt.autoscale(tight=True)

#plotting graph of variables w.r.t target for comparison
t0 = data.loc[data['target']==1]   
t1 = data.loc[data['target'] == 0]
features = data.columns[1:]

def plot_dist1(df1,df2,features,nrows,ncols,label_df1,label_df2):
       fig, axs = plt.subplots(ncols,nrows)
       plt.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.08)
       j = 1
       for feature in features:
              print(feature)
              plt.subplot(nrows,ncols,j)
              sns.distplot(df1[feature], hist = False,label= label_df1)
              sns.distplot(df2[feature],hist = False,label= label_df2)
              j+=1

plot_dist1(t0,t1,features[0:20],4,5,0,1)
plot_dist1(t0,t1,features[20:40],4,5,0,1)
plot_dist1(t0,t1,features[40:60],4,5,0,1)
plot_dist1(t0,t1,features[60:80],4,5,0,1)
plot_dist1(t0,t1,features[80:100],4,5,0,1)
plot_dist1(t0,t1,features[100:120],4,5,0,1)
plot_dist1(t0,t1,features[120:140],4,5,0,1)
plot_dist1(t0,t1,features[140:160],4,5,0,1)
plot_dist1(t0,t1,features[160:180],4,5,0,1)
plot_dist1(t0,t1,features[180:200],4,5,0,1)

#Ploting graph by row, to observe any pattern in variables combining to give 1 and 0
def plot_dist(df1,df2,nrows,ncols):
       fig, axs = plt.subplots(ncols,nrows)
       plt.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.08)
       j=1
       for count in range(df1.shape[0]):
              plt.subplot(nrows,ncols,j)
              sns.distplot(df1.iloc[count,1:], hist = False,label= 0)
              sns.distplot(df2.iloc[count,1:],hist = False,label=1)
              j+=1
plot_dist(t0.iloc[0:20,:],t1.iloc[0:20,:],4,5)
plot_dist(t0.iloc[20:40,:],t1.iloc[20:40,:],4,5)

#################################################################################

"""Model Building"""

def splitter(X,y):
       """fuction for test train spli"""
       X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                        test_size = 0.20,
                                                        random_state=20)
       print("test: \n", y_test.groupby(by=y_test[:]).count())
       print("train: \n", y_train.groupby(by=y_train[:]).count())
       return (X_train,X_test,y_train,y_test)

def smote(X_train,y_train):
       """Function for SMOTE or KNN resampling"""
       sm = SMOTE(random_state=20, ratio = 1.0)
       X_train,y_train= sm.fit_sample(X_train,y_train)
       print("Increasing minority data using SMOTE (synthetic minority data over sampling)."+"\n"+
             "This is to be done only with train data.it’s important to generate the new"+"\n"+
             "samples only in the training set to ensure our model generalizes well to unseen data.")
       #print("train: \n", y_train.groupby(by=y_train[:]).count())
       return (X_train,y_train)

def upsample(X_train,y_train):
       print("Upsampling the minority class \n"+
             "Before upsampling you need to divide the data into traing and test")
       X = pd.concat([X_train, y_train], axis=1)# concatenate our training data back together
       t0 = X[X.target==0]#seperating all variables w.r.t. 0
       t1 = X[X.target==1]#seperating all variables w.r.t. 1
       upsampled = resample(t1,
                            replace=True, # sample with replacement
                            n_samples=len(t0), # match number in majority class
                            random_state=10) # reproducible results
       upsampled = pd.concat([t0, upsampled])
       y_train = upsampled['target']
       X_train = upsampled.drop('target',axis=1)
       print("train: \n", y_train.groupby(by=y_train[:]).count())
       return (X_train,y_train)


def logistic(X_train,y_train):
       classifier= LogisticRegression(random_state=0)
       classifier.fit(X_train, y_train)
       return classifier

def knn(X_train,y_train,n_neighbors):
       classifier = KNeighborsClassifier(n_neighbors=n_neighbors,metric='minkowski',p=2)
       classifier.fit(X_train,y_train)
       return classifier

def tree(X_train,y_train):
       classifier = DecisionTreeClassifier(random_state = 20,
                                           criterion = 'entropy')
       classifier.fit(X_train,y_train)
       return classifier

def randomforest(n_estimators,criterion,X_train,y_train):
       """Function for Random Forest"""
       classifier = RandomForestClassifier(n_estimators = n_estimators,
                                           criterion = criterion,
                                           random_state=20)
       classifier.fit(X_train,y_train)
       return classifier

def naive(X_train,y_train):
       classifier = GaussianNB()
       classifier.fit(X_train,y_train)
       return classifier

def pred_cm(X_test,y_test,classifier):
       """Function for classifier prediction and confusion matrix"""
       y_pred = classifier.predict(X_test)
       cm = confusion_matrix(y_test,y_pred)
       TN,FP,FN,TP = cm[0,0],cm[0,1],cm[1,0],cm[1,1]
       print("Accuracy = ",round((TN+TP)/len(y_pred),3))
       print("Recall = ",round(TP/(TP+FN),3))
       print("Precision = ", round(TP/(TP+FP),3))
       return (y_pred,cm)

def auc(y_test,y_pred):
       print("AUC Score = ",roc_auc_score(y_test,y_pred)) 
       fpr, tpr, _ = roc_curve(y_test, y_pred)
       plt.clf()
       plt.plot(fpr, tpr)
       plt.xlabel('FPR')
       plt.ylabel('TPR')
       plt.title('ROC curve')
       plt.show()

#Splitting depenent variable and independent variable
y = data.target
X = data.drop(['target'],axis = 1)

########################----Model set 01----#################################
"""All models in this set are based on normal unteweked dataset"""
#Splitting data into train and test 
X_train,X_test,y_train,y_test=splitter(X,y)

# Model 01 Logistic Regression
classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.914
#Recall = 0.26
#Precision = 0.679
#AUC Score = 0.624


#model 02 Naive Bayes
classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.921
#Recall = 0.3661
#Precision = 0.70
#AUC Score = 0.6744


#model 03 Random Forest
"""In this model the higher the n_estimators the poorer is prediction"""
classifier=randomforest(10,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.829
#Recall = 0.175
#Precision = 0.164
#AUC Score = 0.538


#model 04 Decision Tree
classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.837
#Recall = 0.199
#Precision = 0.19
#AUC Score = 0.553


#Normalisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
#Model 05 KNN
classifier = knn(X_train,y_train,5)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.899
#Recall = 0.002
#Precision = 0.166
#AUC Score = 0.500


########################----Model set 02----#################################
"""In this odel we are increasing the minority class by SMOTE"""
#SMOTE
y = data.target
X = data.drop(['target'],axis = 1)
X_train,y_train = smote(X_train,y_train)

# Model 01 logistic Regression
classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.787
#Recall = 0.771
#Precision = 0.288
#AUC Score = 0.780


#model 02 Naive Bayes
classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.869
#Recall = 0.083
#Precision = 0.173
#AUC Score = 0.519


#model 03 Random Forest
"""In this model the higher the n_estimators the poorer is prediction"""
classifier=randomforest(3,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.746
#Recall = 0.127
#Precision = 0.263
#AUC Score = 0.531
2

#model 04 Decision Tree
classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.705
#Recall = 0.31
#Precision = 0.12
#AUC Score = 0.532


#Model 05 KNN
#Normalisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

classifier = knn(X_train,y_train,3)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.108
#Recall = 0.99
#Precision = 0.099
#AUC Score = 0.501



########################----Model set 03----#################################
"""Upsampling the dataset"""
#Splitting again
y = data.target
X = data.drop(['target'],axis = 1)
#Splitting data into train and test 
X_train,X_test,y_train,y_test=splitter(X,y)

#upsample
X_train,y_train = upsample(X_train,y_train)

# Model 01 logistic Regression
classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.782
#Recall = 0.775
#Precision = 0.283
#AUC Score = 0.779


#model 02 Naive Bayes
classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.813
#Recall = 0.795
#Precision = 0.322
#AUC Score = 0.805


#model 03 Random Forest
"""In this model the higher the n_estimators the poorer is prediction"""
classifier=randomforest(3,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.865
#Recall = 0.135
#Precision = 0.218
#AUC Score = 0.540


#model 04 Decision Tree
classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.829
#Recall = 0.171
#Precision = 0.162
#AUC Score = 0.536

"""As there is no such improvement in KNN so we will not consider it from now"""

########################----Model set 04----#################################
"""Dropping few variables from the dataset as those variables conatins same value for 
0 and 1. Data will be upsampled as the best result was seen in upsampled dataset"""
y = data.target
X = data.drop(['target'],axis = 1)
#Dropping variables
drop_variables = ['var_17','var_38','var_46','var_59','var_61','var_65',
                  'var_73','var_79','var_96','var_100','var_124','var_185']
for variable in drop_variables:
       X=X.drop(variable,axis=1)
    
#Splitting data into train and test 
X_train,X_test,y_train,y_test=splitter(X,y)

#upsample
X_train,y_train = upsample(X_train,y_train)

# Model 01 logistic Regression
classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.781
#Recall = 0.775
#Precision = 0.282
#AUC Score = 0.778


#model 02 Naive Bayes
classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.812
#Recall = 0.794
#Precision = 0.321
#AUC Score = 0.804


#model 03 Random Forest
"""In this model the higher the n_estimators the poorer is prediction"""
classifier=randomforest(3,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.866
#Recall = 0.132
#Precision = 0.219
#AUC Score = 0.540


#model 04 Decision Tree
classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.831
#Recall = 0.171
#Precision = 0.165
#AUC Score = 0.537


########################----Model set 05----#################################
"""Dropping few variables from the dataset as those variables conatins same value for 
0 and 1. Data will be SMOTE as the best result was seen in upsampled dataset"""
y = data.target
X = data.drop(['target'],axis = 1)
#Dropping variables
drop_variables = ['var_17','var_38','var_46','var_59','var_61','var_65',
                  'var_73','var_79','var_96','var_100','var_124','var_185']
for variable in drop_variables:
       X=X.drop(variable,axis=1)
       
#Splitting data into train and test 
X_train,X_test,y_train,y_test=splitter(X,y)

#upsample
X_train,y_train = smote(X_train,y_train)

# Model 01 logistic Regression
classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.787
#Recall = 0.769
#Precision = 0.287
#AUC Score = 0.779


#model 02 Naive Bayes
classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.866
#Recall = 0.10
#Precision = 0.183
#AUC Score = 0.525


#model 03 Random Forest
"""In this model the higher the n_estimators the poorer is prediction"""
classifier=randomforest(3,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.746
#Recall = 0.272
#Precision = 0.138
#AUC Score = 0.535


#model 04 Decision Tree
classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.703
#Recall = 0.319
#Precision = 0.122
#AUC Score = 0.532




########################----Model set 06----#################################
"""Trying some feature enginering with normalization.
We will include:
       Square of each row of each column
       cube of each row of each column"""
y = data.target
X = data.drop(['target'],axis = 1)
for col in X.columns:
       X[col+'_sq'] = X[col]*X[col]
       X[col+'-cu'] = X[col]*X[col]*X[col]

#Splitting data into train and test 
X_train,X_test,y_train,y_test=splitter(X,y)
#upsample
X_train,y_train = upsample(X_train,y_train)
#Normalisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Model 01 logistic Regression
classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy =  0.871
#Recall =  0.349
#Precision =  0.696
#AUC score = 0.66

# Model 02 NAive Bayes
classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)

#Accuracy =  0.611
#Recall =  0.904
#Precision =  0.192
#AUC Score = 0.741

########################----Model set 07----#################################
"""Trying some feature enginering with normalization.
We will include:
       Square of each row of each column
       rank of each variable values
       cube of each row of each column"""
y = data.target
X = data.drop(['target'],axis = 1)
for col in X.columns:
       X[col+'_sq'] = X[col]*X[col]
       X[col+'-cu'] = X[col]*X[col]*X[col]
       X[col+'ra'] = rankdata(X[col]).astype('float32')

#Splitting data into train and test 
X_train,X_test,y_train,y_test=splitter(X,y)
#upsample
X_train,y_train = upsample(X_train,y_train)
#Normalisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Model 01 logistic Regression
classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy =  0.871
#Recall =  0.349
#Precision =  0.696
#AUC score = 0.66

# Model 02 NAive Bayes
classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#upsample + sclae
#Accuracy =  0.798
#Recall =  0.684
#Precision =  0.286
#AUC = 0.7472242170114228



#################################################################################
"""Plotting model output"""
set1 = [[0.914, 0.921, 0.829, 0.837, 0.899],
        [0.26,0.366,0.175,0.199,0.002],
        [0.679,0.70,0.164,0.19,0.166],
        [0.624, 0.674, 0.538, 0.553, 0.5]]
set2 = [[0.787,0.869,0.746,0.705,0.108],
        [0.771,0.083,0.127,0.31,0.99],
        [0.288,0.173,0.263,0.12,0.099],
        [0.780,0.519,0.531,0.532,0.501]]
set3 = [[0.782,0.813,0.865,0.829,0],
         [0.775,0.795,0.135,0.171,0],
         [0.283,0.322,0.218,0.162,0],
         [0.779,0.805,0.540,0.536,0]]
set4 = [[0.781, 0.812, 0.866, 0.831,0],
        [0.775, 0.794, 0.132, 0.171,0],
        [0.282, 0.321, 0.219, 0.165,0],
        [0.778, 0.804, 0.54, 0.537,0]]
set5 = [[0.787,0.866,0.746,0.703,0],
        [0.769,0.10,0.272,0.319,0],
        [0.287,0.183,0.138,0.1222,0],
        [0.779,0.525,0.535,0.532,0]]
set6 = [[0.871, 0.611, 0, 0, 0],
        [0.349, 0.904, 0, 0, 0],
        [0.696, 0.192, 0, 0, 0],
        [0.66, 0.741, 0, 0, 0]]
set7 = [[0.871, 0.798, 0, 0, 0],
        [0.349, 0.684, 0, 0, 0],
        [0.696, 0.286, 0, 0, 0],
        [0.66, 0.747, 0, 0, 0]]
models = ['logistic','naive','random','tree','knn']
r = np.arange(5)
plt.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.08)
fig1 = plt.subplot(3,3,1)
fig1.bar(r,set1[0],label='accuracy',color = 'gray',width = 0.10)
fig1.bar(r+0.10,set1[1],label='recall',color='#03fcce',width=0.10)
fig1.bar(r+0.20,set1[2],label='precision',color='pink',width=0.10)
fig1.bar(r+0.30,set1[3],label='AUC score',color='#fcba03',width=0.10)
fig1.legend()
plt.title("Raw model",loc='left')
plt.xticks([model for model in range(len(models))],models)


fig2 = plt.subplot(3,3,2)
fig2.bar(r,set2[0],label='accuracy',color = 'gray',width = 0.10)
fig2.bar(r+0.10,set2[1],label='recall',color='#03fcce',width=0.10)
fig2.bar(r+0.20,set2[2],label='precision',color='pink',width=0.10)
fig2.bar(r+0.30,set2[3],label='AUC score',color='#fcba03',width=0.10)
fig2.legend()
plt.title("SMOTE model",loc='left')
plt.xticks([model for model in range(len(models))],models)


fig3 = plt.subplot(3,3,3)
fig3.bar(r,set3[0],label='accuracy',color = 'gray',width = 0.10)
fig3.bar(r+0.10,set3[1],label='recall',color='#03fcce',width=0.10)
fig3.bar(r+0.20,set3[2],label='precision',color='pink',width=0.10)
fig3.bar(r+0.30,set3[3],label='AUC score',color='#fcba03',width=0.10)
fig3.legend()
plt.title("Upsampling model",loc='left')
plt.xticks([model for model in range(len(models))],models)


fig4 = plt.subplot(3,3,4)
fig4.bar(r,set4[0],label='accuracy',color = 'gray',width = 0.10)
fig4.bar(r+0.10,set4[1],label='recall',color='#03fcce',width=0.10)
fig4.bar(r+0.20,set4[2],label='precision',color='pink',width=0.10)
fig4.bar(r+0.30,set4[3],label='AUC score',color='#fcba03',width=0.10)
fig4.legend()
plt.title("Variable drop upsample model",loc='left')
plt.xticks([model for model in range(len(models))],models)


fig5 = plt.subplot(3,3,5)
fig5.bar(r,set5[0],label='accuracy',color = 'gray',width = 0.10)
fig5.bar(r+0.10,set5[1],label='recall',color='#03fcce',width=0.10)
fig5.bar(r+0.20,set5[2],label='precision',color='pink',width=0.10)
fig5.bar(r+0.30,set5[3],label='AUC score',color='#fcba03',width=0.10)
fig5.legend()
plt.title("Variable drop SMOTE model",loc='left')
plt.xticks([model for model in range(len(models))],models)

fig6 = plt.subplot(3,3,6)
fig6.bar(r,set6[0],label='accuracy',color = 'gray',width = 0.10)
fig6.bar(r+0.10,set6[1],label='recall',color='#03fcce',width=0.10)
fig6.bar(r+0.20,set6[2],label='precision',color='pink',width=0.10)
fig6.bar(r+0.30,set6[3],label='AUC score',color='#fcba03',width=0.10)
fig6.legend()
plt.title("Feature Enginnering 01",loc='left')
plt.xticks([model for model in range(len(models))],models)

fig7 = plt.subplot(3,3,7)
fig7.bar(r,set7[0],label='accuracy',color = 'gray',width = 0.10)
fig7.bar(r+0.10,set7[1],label='recall',color='#03fcce',width=0.10)
fig7.bar(r+0.20,set7[2],label='precision',color='pink',width=0.10)
fig7.bar(r+0.30,set7[3],label='AUC score',color='#fcba03',width=0.10)
fig7.legend()
plt.title("Feature Enginnering 02",loc='left')
plt.xticks([model for model in range(len(models))],models)

#Creating Submission file
y = data.target
X = data.drop(['target'],axis = 1)
X_train,X_test,y_train,y_test=splitter(X,y)
X_train,y_train = upsample(X_train,y_train)
classifier = naive(X_train,y_train)
test_pred=classifier.predict(test)
test_pred_list = list(test_pred)
results = pd.DataFrame({'target':test_pred_list})
results['ID_code'] = testID
results = results[['ID_code','target']]
results.to_csv('submission_Recall_Precision.csv',header=True,index=False)
test_pred_proba=classifier.predict_proba(test)
results['target'] = test_pred_proba[:,1]
results.to_csv('submission_AUC.csv',header=True,index=False)

###############################################################################################################
