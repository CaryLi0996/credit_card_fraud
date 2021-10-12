#!/usr/bin/env python
# coding: utf-8

# # dataset is from https://www.kaggle.com/mlg-ulb/creditcardfraud

# In[1]:


# Import necessary packages 

import pandas as pd 
import numpy as np


# In[2]:


# Import the dataset as a dataframe

credit_trans = pd.read_csv('creditcard.csv')
credit_trans.head()


# In[3]:


# Drop the unuseful time column

credit_trans.drop('Time', axis=1, inplace=True)
credit_trans.head()


# In[4]:


# Check the number of transcations in the two classes

total_credit_trans = len(credit_trans)
non_fraud_trans = len(credit_trans[credit_trans['Class'] == 0])
fraud_trans = len(credit_trans[credit_trans['Class'] == 1])
percentage_non_fraud = round(non_fraud_trans / total_credit_trans * 100, 2)
percentage_fraud = round(fraud_trans / total_credit_trans * 100, 2)

print('the total numebr of credit card transactions is {}'.format(total_credit_trans))
print('the number of non-fraudulent transactions is {}'.format(non_fraud_trans))
print('the number of fraudulent transactions is {}'.format(fraud_trans))
print('the percentage of non-fraudulent transactions out of all transcations is {}'.format(percentage_non_fraud))
print('the percentage of fraudulent transactions is {}'.format(percentage_fraud))


# In[5]:


# Splitting the dataset 

from sklearn.model_selection import train_test_split

X = credit_trans.drop('Class', axis = 1)
y = credit_trans['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[6]:


# Try to balance this extremely unbalanced dataset before modeling 

# Method: oversample the fraudulent cases in the train sets, keeping the test sets unchanged

from sklearn.utils import resample 

train_set = pd.concat([X_train, y_train], axis = 1)

train_non_fraud = train_set[train_set['Class'] == 0]
train_fraud = train_set[train_set['Class'] == 1]


# In[7]:


# Now the train set is balanced after applying oversampling method on the original train set

oversampled_fraud = resample(train_fraud, replace = True, n_samples = len(train_non_fraud), random_state = 1)

oversampled_train_set = pd.concat([train_non_fraud, oversampled_fraud])

oversampled_train_set['Class'].value_counts()


# In[8]:


# reassign the training set after oversampling 

X_train = oversampled_train_set.drop('Class', axis = 1)
y_train = oversampled_train_set['Class']


# In[9]:


# Modeling for binary classification: 

# since this is a binary classification task, and our objective is to choose the model that has the best performance against our performance metric in classifying fraudulent transactions in unseen dataset. 

# The following models are suitable in this case and I will compare them by different evaluation standards:  


# 1. logistic regression
# 2. knn
# 3. svm
# 4. decision tree
# 5. random forest
# 6. bagging
# 7. boosting 


# In[10]:


# logistic regression model

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

print("train set score: " + str(lr.score(X_train, y_train)))
print("test set score: " + str(lr.score(X_test, y_test)))


# In[11]:


# knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier (n_neighbors = 10)
knn.fit(X_train, y_train)

print("train set score: " + str(knn.score(X_train, y_train)))
print("test set score: " + str(knn.score(X_test, y_test)))


# In[15]:


# support vector machine model 

from sklearn.svm import LinearSVC

svm = LinearSVC(C=0.0001)
svm.fit(X_train, y_train)

print("train set score: " + str(svm.score(X_train, y_train)))
print("test set score: " + str(svm.score(X_test, y_test)))


# In[13]:


# decision tree 

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

print("train set score: " + str(clf.score(X_train, y_train)))
print("test set score: " + str(clf.score(X_test, y_test)))


# In[17]:


# random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 29, max_depth = 10)
rfc.fit(X_train, y_train)

print("train set score: " + str(rfc.score(X_train, y_train)))
print("test set score: " + str(rfc.score(X_test, y_test)))


# In[18]:


# bagging

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bc = BaggingClassifier(DecisionTreeClassifier(), max_samples = 0.7, max_features = 1.0, n_estimators = 29)
bc.fit(X_train, y_train)

print("train set score: " + str(bc.score(X_train, y_train)))
print("test set score: " + str(bc.score(X_test, y_test)))


# In[21]:


# boosting 

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

adb = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=12, max_depth=8), n_estimators=29, learning_rate = 0.6)
adb.fit(X_train, y_train)

print("train set score: " + str(adb.score(X_train, y_train)))
print("test set score: " + str(adb.score(X_test, y_test)))


# ## Boosting model generates the best accuracy score in predicting credit card fraud

# In[ ]:




