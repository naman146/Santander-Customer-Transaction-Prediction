#!/usr/bin/env python3

print("The dataset both train and test countains all numeric values. There is "+ 
"no description of the varibale label. There is no NA value in the dataset "+
"both test and train. Exploring the train data we found that about 89% of "+
"of the data countains 0(zero) as the target variable and only 11% countains "+ 
"1. So we nned to sample out data or else the model will be baised towards 0")
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

print("Importing dataset")
#os.chdir('D:\MY PROGRAMMING DATA\edwisor\Project01')
data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#################################################################################
print("Exploring dataset")

print("HEAD Train \n ",data.head())
print("HEAD Test \n ",test.head())
print("Describe Train \n ",data.describe())
print("Find NA \n ",data.isna().sum())
print("Find NA \n ",test.isna().sum()) #there is no NA in the dataset train and test
#Dropping ID_code variable and changing index to ID_code
data.index = data.ID_code
data = data.drop('ID_code',axis=1)
test.index = test.ID_code
testID = test['ID_code']
test = test.drop('ID_code',axis=1)
#Finding correlation
print("Correlation Dataframe \n ",data.corr())
#from the correlation heatmap we can see that there is no high correlation between
#any two variables.
#normal distribution of data/hyporthesis test
"""Shapiro-Wilk test can be performed as follow:
       Null hypothesis: the data are normally distributed
       Alternative hypothesis: the data are not normally distributed"""
print("P-value is ",shapiro(data['var_0'][0:5000]))#p-value 4.205e-13
print("P-value is ",shapiro(data['var_115'][0:5000]))#p-value 1.258e-08

#################################################################################
"""Model Building"""
print("Model Building")

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
print("########################----Model set 01----#################################")
"""All models in this set are based on normal unteweked dataset"""
print("All models in this set are based on normal unteweked dataset")

#Splitting data into train and test 
X_train,X_test,y_train,y_test=splitter(X,y)

# Model 01 Logistic Regression
print("Model 01 Logistic Regression")
classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.914
#Recall = 0.26
#Precision = 0.679
#AUC Score = 0.624
#cm = 35525,493,
     #2939,1043

#model 02 Naive Bayes
print("#model 02 Naive Bayes")
classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.921
#Recall = 0.3661
#Precision = 0.70
#AUC Score = 0.6744
#cm = 35396,622,
     #2524,1458

#model 03 Random Forest
print("#model 03 Random Forest")
"""In this model the higher the n_estimators the poorer is prediction"""
classifier=randomforest(10,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.829
#Recall = 0.175
#Precision = 0.164
#AUC Score = 0.538
#cm = 32470,3548,
      #3283,699

#model 04 Decision Tree
print("#model 04 Decision Tree")
classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.837
#Recall = 0.199
#Precision = 0.19
#AUC Score = 0.553
#cm = 32704,3314,
      #3189,793

#Model 05 KNN
print("Model 05 KNN")
#Normalisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

classifier = knn(X_train,y_train,5)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.899
#Recall = 0.002
#Precision = 0.166
#AUC Score = 0.500
#cm = 35963,55,
      #3971,11

########################----Model set 02----#################################
print("########################----Model set 02----#################################")

"""In this odel we are increasing the minority class by SMOTE"""
#SMOTE
y = data.target
X = data.drop(['target'],axis = 1)
X_train,y_train = smote(X_train,y_train)

# Model 01 logistic Regression
print("Model 01 Logistic Regression")

classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.787
#Recall = 0.771
#Precision = 0.288
#AUC Score = 0.780


#model 02 Naive Bayes
print("#model 02 Naive Bayes")

classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.869
#Recall = 0.083
#Precision = 0.173
#AUC Score = 0.519


#model 03 Random Forest
"""In this model the higher the n_estimators the poorer is prediction"""
print("#model 03 Random Forest")
classifier=randomforest(3,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.746
#Recall = 0.127
#Precision = 0.263
#AUC Score = 0.531


#model 04 Decision Tree
print("#model 04 Decision Tree")

classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.705
#Recall = 0.31
#Precision = 0.12
#AUC Score = 0.532

#Model 05 KNN
print("Model 05 KNN")
#Normalisation
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

classifier = knn(X_train,y_train,5)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.108
#Recall = 0.99
#Precision = 0.099
#AUC Score = 0.501



########################----Model set 03----#################################
print("########################----Model set 03----#################################")
"""Upsampling the dataset"""
#Splitting again
y = data.target
X = data.drop(['target'],axis = 1)
#Splitting data into train and test 
X_train,X_test,y_train,y_test=splitter(X,y)

#upsample
X_train,y_train = upsample(X_train,y_train)

# Model 01 logistic Regression
print("Model 01 Logistic Regression")

classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.782
#Recall = 0.775
#Precision = 0.283
#AUC Score = 0.779

#model 02 Naive Bayes
print("#model 02 Naive Bayes")

classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.813
#Recall = 0.795
#Precision = 0.322
#AUC Score = 0.805


#model 03 Random Forest
"""In this model the higher the n_estimators the poorer is prediction"""
print("#model 03 Random Forest")
classifier=randomforest(3,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.865
#Recall = 0.135
#Precision = 0.218
#AUC Score = 0.540


#model 04 Decision Tree
print("#model 04 Decision Tree")

classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.829
#Recall = 0.171
#Precision = 0.162
#AUC Score = 0.536


"""As there is no such improvement in KNN so we will not consider it from now"""

########################----Model set 04----#################################
print("########################----Model set 04----#################################")
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
print("#model 01 logistic Regression")

classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.781
#Recall = 0.775
#Precision = 0.282
#AUC Score = 0.778


#model 02 Naive Bayes
print("#model 02 Naive Bayes")

classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.812
#Recall = 0.794
#Precision = 0.321
#AUC Score = 0.804


#model 03 Random Forest
"""In this model the higher the n_estimators the poorer is prediction"""
print("#model 03 Random Forest")
classifier=randomforest(3,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.866
#Recall = 0.132
#Precision = 0.219
#AUC Score = 0.540


#model 04 Decision Tree
print("#model 04 Decision Tree")

classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.831
#Recall = 0.171
#Precision = 0.165
#AUC Score = 0.537


########################----Model set 05----#################################
print("########################----Model set 05----#################################")
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
print("Model 01 Logistic Regression")

classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.787
#Recall = 0.769
#Precision = 0.287
#AUC Score = 0.779


#model 02 Naive Bayes
print("#model 02 Naive Bayes")

classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.866
#Recall = 0.10
#Precision = 0.183
#AUC Score = 0.525


#model 03 Random Forest
"""In this model the higher the n_estimators the poorer is prediction"""
print("#model 03 Random Forest")
classifier=randomforest(3,'entropy',X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.746
#Recall = 0.272
#Precision = 0.138
#AUC Score = 0.535


#model 04 Decision Tree
print("#model 04 Decision Tree")

classifier = tree(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy = 0.703
#Recall = 0.319
#Precision = 0.122
#AUC Score = 0.532



########################----Model set 06----#################################
print("########################----Model set 06----#################################")
"""Trying some feature enginering with normalization.
We will include:
       Square of each row of each column
       value - maens of that columns
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
print("Model 01 Logistic Regression")

classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy =  0.871
#Recall =  0.349
#Precision =  0.696
#AUC score = 0.66

# Model 02 NAive Bayes
print("#model 02 Naive Bayes")

classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)

#Accuracy =  0.611
#Recall =  0.904
#Precision =  0.192
#AUC Score = 0.741

########################----Model set 07----#################################
print("########################----Model set 07----#################################")
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
print("Model 01 Logistic Regression")

classifier = logistic(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#Accuracy =  0.871
#Recall =  0.349
#Precision =  0.696
#AUC score = 0.66

# Model 02 NAive Bayes
print("#model 02 Naive Bayes")

classifier = naive(X_train,y_train)
y_pred,cm = pred_cm(X_test,y_test,classifier)
auc_score = auc(y_test,y_pred)
#upsample + sclae
#Accuracy =  0.798
#Recall =  0.684
#Precision =  0.286
#AUC = 0.7472242170114228

###############################################################################################################
print("writing file")

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