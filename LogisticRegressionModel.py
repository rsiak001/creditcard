#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Import dataset
import pandas
df = pandas.read_csv("CreditCard.csv")
print(df)
#noted there are 3428 rows of data and 4 columns


# Data Wrangling

# In[22]:


df.boxplot('income')
#no outliers identified for income


# In[23]:


df.boxplot('age')
#noted there are 3 outliers that are in negative range for 'age', to be removed under the data cleaning segment


# In[24]:


df.boxplot('loan')
#Noted that there are no outliers in loan variable.


# In[25]:


df.boxplot('default')
#noted there are no outleirs in default variable


# Data Cleaning

# In[26]:


#Remove negative values of age 

#Identifying Outliers with Interquartile Range (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[27]:


#Finding the interquartile range for "age" and the upper and lower bounds
import numpy as np
Q1 = np.percentile(df["age"],25, interpolation = "midpoint")
Q3 = np.percentile(df["age"],75, interpolation = "midpoint")
IQR = Q3 - Q1
upper = np.where(df["age"] >= (Q3 + 1.5*IQR))
lower = np.where(df["age"] <= (Q1 - 1.5*IQR))


# In[28]:


#Drop outliers above Q3 and below Q1
df.drop(upper[0],inplace = True)
df.drop(lower[0],inplace = True)
df
#Revised number of rows are 3425, suggesting the 3 outliers have been removed


# In[29]:


#Remove values that are not a number
for i in df.columns:
    df1 = pandas.to_numeric(df[i], errors='coerce')
    df=df[df1.notnull()] #make it to null then remove null
print(df)
#Since there are 3425 rows, suggest there are no values that are not a number in the dataset


# In[30]:


import matplotlib.pyplot as plt
df.hist()


# In[31]:


#Check Correlation - Check co-linearity, the relationship between the independent variable
df.corr()
import seaborn as sns
sns.heatmap(df.corr())


# In[32]:


#Feature Selection
#We do not do feature selection, as the model will have a built-inselection features.

#Select Kbest only when want to select 5 out of 100 features.
#Due to the limited number of features of 3, will not perform feature selection

#Normalisation
#There is also no normalisation because normalisation usually requires to find coefficient within the fastest time.
#However, since there are only 3 X variables, the model will converge quickly with about 15-16 convergements.

#Logistic Regression
#Perform Logistic Regression because Y axis is categorical


# In[33]:


#1. Check Collinearity of the X variables
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)

#All 3 variables are greater than 5, suggesting that they are highly correlated.
#However we do not remove them as there are only 3 x variables in the data set.
#Due to the limited number of features of 3, we will not perform feature selection


# In[17]:


#2. Perform Train-test split
X = df.loc[:,['age','income','loan']]
Y = df.loc[:,['default']]
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)
print(X_train, X_test, Y_train, Y_test )


# In[34]:


#3. Perform Logistic Regression for train-set

#Predict model of train-set
from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_train)


#Obtain confusion matrix and train-set accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
#[[1107   84]
#[  38 1168]]
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)
#0.9491030454735085


# In[35]:


#4. Perform Logistic Regression for test-set
pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
#[[487  33]
#[ 20 488]]
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
testset_accuracy = print(accuracy)
#0.9484435797665369


# In[36]:


#Perform QQPlot to check heteroskedasticity
import statsmodels.api as sm
from matplotlib import pyplot as plt
mod_fit = sm.OLS(Y,X).fit()
res = mod_fit.resid # residuals
fig = sm.qqplot(res)
plt.show()


# In[37]:


#Perform Anova on Logistic Model
import statsmodels.api as sm
model=sm.Logit(Y,X)
result=model.fit()
print(result.summary2())

