#!/usr/bin/env python
# coding: utf-8

# Decision Tree Model

# In[21]:


#Data cleaning & Importing dataset
##Please refer to LogisticRegression Model for inclusions on data wrangling & data visualisation
#Import dataset
import pandas
df = pandas.read_csv("CreditCard.csv")
print(df)
#noted there are 3428 rows of data and 4 columns

#Remove negative values of age identified

#Identifying Outliers with Interquartile Range (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

#Identifying the interquartile range for "age" and the upper and lower bounds
import numpy as np
Q1 = np.percentile(df["age"],25, interpolation = "midpoint")
Q3 = np.percentile(df["age"],75, interpolation = "midpoint")
IQR = Q3 - Q1
upper = np.where(df["age"] >= (Q3 + 1.5*IQR))
lower = np.where(df["age"] <= (Q1 - 1.5*IQR))

##Drop outliers above Q3 and below Q1
df.drop(upper[0],inplace = True)
df.drop(lower[0],inplace = True)
df
#Revised number of rows are 3425, suggesting the 3 outliers have been removed

##Remove values that are not a number
for i in df.columns:
    df1 = pandas.to_numeric(df[i], errors='coerce')
    df=df[df1.notnull()] #make it to null then remove null
print(df)
#Since there are 3425 rows, suggest there are no values that are not a number in the dataset


# In[22]:


#1. Import DecisionTreeClassifier as Y is continuous
from sklearn import tree
model = tree.DecisionTreeClassifier()


# In[23]:


#2. Define X and Y variables before performing Train Test Split
X = df.loc[:,['age','income','loan']]
Y = df.loc[:,['default']]
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)


# In[24]:


#3. Predict using train set and print accuracy of train-set
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
pred = model.predict(X_train)

#Obtain confusion matrix and train-set accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
#[[1185    0]
# [   0 1212]]

accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)
#1.0


# In[25]:


#4. Predict using test set and print accuracy of test-set
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
#[[515  11]
#[  0 502]]
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)
#0.9892996108949417


# In[26]:


#5. Use Grid (max depth)

#Find max depth
import math
from sklearn.model_selection import GridSearchCV

model = tree.DecisionTreeClassifier()
grid = GridSearchCV(estimator = model, param_grid = dict(max_depth = [i for i in range(1, 20)]))
grid_results = grid.fit(X, Y)
grid_results.best_params_

#{'max_depth': 14}


# In[27]:


#6. Optimization using Grid (min sample split)
import math
from sklearn.model_selection import GridSearchCV

model = tree.DecisionTreeClassifier()
grid = GridSearchCV(estimator = model, param_grid = dict( min_samples_split = [i for i in range(3, 20)]))
grid_results = grid.fit(X, Y)
grid_results.best_params_

#{'min_samples_split': 12}

