#!/usr/bin/env python
# coding: utf-8

# Neural Network Model

# In[5]:


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


# In[6]:


#1. Perform normalisation of x variables before train-test split 
X = df.loc[:,['age','income','loan']]
Y = df.loc[:,['default']]
from scipy import stats
import numpy as np 
import pandas as pd
pd.set_option('display.max_rows', 10)
for i in X.columns:
    X[i]=stats.zscore(X[i].astype(np.float))
print(X)


# In[7]:


#2. Perform train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
print(X_train, X_test, Y_train, Y_test)


# In[8]:


#3. Keras Sequential 

from keras.models import Sequential
from keras.layers import Dense, Dropout

model=Sequential()


# In[9]:


#4.Network Design
model.add(Dense(10, input_dim=3, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(8, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))


# In[10]:


#5.Compile (Configuration)
model.compile(loss = 'binary_crossentropy', optimizer = 'Adamax', metrics = ['accuracy'])

#Did not compile optimizer for the NNet Model as the system will auto optimize


# In[11]:


#6.Fit
model.fit(X_train, Y_train, batch_size = 10, epochs=15, verbose=1)
#240/240 [==============================] - 0s 2ms/step - loss: 0.1740 - accuracy: 0.9266


# In[13]:


#7. Evaluate train-set model
model.evaluate(X_train, Y_train)
#0.9804


# In[14]:


#8. Evaluate test-set model
model.evaluate(X_test, Y_test)
#0.9815


# In[15]:


#9. Prediction for NNet model
import numpy as np
from sklearn.metrics import confusion_matrix


# In[16]:


#10.Predict Neural Network Model using confusion matrix & Obtain accuracy for Train set
pred=model.predict(X_train)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Y_train, pred)
print(cm)
#[[1121   81]
#[   9 1186]]
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)
#0.9624530663329162


# In[17]:


#11.Predict Neural Network Model using confusion matrix & Obtain accuracy for Test set
pred=model.predict(X_test)
pred=np.where(pred>0.5,1,0)
cm=confusion_matrix(Y_test, pred)
print(cm)
#[[472  37]
#[  3 516]]
accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)
#0.9610894941634242


# In[18]:


#12. Summary of Model
model.summary()

#13. Summary of Weight
model.weights


# In[19]:


#14. Import Load from joblib
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(6,2))
#Limited-memory Broyden–Fletcher–Goldfarb–Shanno

model.fit(X, Y)
pred = model.predict(X)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, pred)
print(cm)


# In[20]:


#15. Dump external file & load
import joblib
joblib.dump(model,"NeuralNetworkModel")


# In[ ]:




