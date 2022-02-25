#!/usr/bin/env python
# coding: utf-8

# For Neural Network Model

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import request, render_template
import joblib


# In[4]:


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




