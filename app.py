#!/usr/bin/env python
# coding: utf-8

# For Neural Network Model

# In[6]:


from flask import Flask


# In[7]:


app = Flask(__name__)


# In[8]:


from flask import request, render_template
import joblib


# In[9]:


@app.route("/", methods = ["GET","POST"])
def index():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        print(income, age, loan)
        model=joblib.load("NeuralNetworkModel")
        pred=model.predict([[float(income),float(age),float(loan)]])
        print(pred)
        s="The predicted default score is:"+str(pred)
        return(render_template("index.html",result=s))
    else:
        return(render_template("index.html",result="2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




