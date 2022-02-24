#!/usr/bin/env python
# coding: utf-8

# XGBoost Model

# In[1]:


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


# In[2]:


#1. Import GradientBoostingClassifier as Y is continuous variable
from sklearn.ensemble import GradientBoostingClassifier


# In[3]:


#2. Define X and Y variables before performing Train Test Split
X = df.loc[:,['age','income','loan']]
Y = df.loc[:,['default']]
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)


# In[5]:


#3. Predict model accuracy on train set and print confusion matrix
model = GradientBoostingClassifier(max_depth=3)
model.fit(X_train, Y_train)
pred = model.predict(X_train)


# In[6]:


#4. Predict SKLearn model using confusion matrix & Obtain accuracy for Train set
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
#[[1192    2]
#[   0 1203]]
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)
#0.999165623696287


# In[7]:


#5. Predict SKLearn model using confusion matrix & Obtain accuracy for Test set
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
#[[501  16]
#[  0 511]]
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)
#0.9844357976653697

