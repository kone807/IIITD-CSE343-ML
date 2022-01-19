#!/usr/bin/env python
# coding: utf-8

# # ML Assignment-2 | Q1 | 2019040

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


def train_test_val_split(x,y,test_ratio,val_ratio):
    
    np.random.seed(0)
    random1 = np.random.rand(x.shape[0])
    split1 = random1 < np.percentile(random1,(1-test_ratio-val_ratio)*100)
    
    x_train = x[split1]
    y_train = y[split1]
    
    x_test_val = x[~split1]
    y_test_val = y[~split1]
    
    random2 = np.random.rand(x_test_val.shape[0])
    
    val_ratio = val_ratio/(val_ratio+test_ratio)
    split2 = random2 < np.percentile(random2,(1-val_ratio)*100)
    
    x_test = x_test_val[split2]
    y_test = y_test_val[split2]
    
    x_val = x_test_val[~split2]
    y_val = y_test_val[~split2]
    
    return x_train,x_test,x_val,y_train,y_test,y_val
    
def accuracy(y_pred, y_test):
    
    return np.sum(y_pred==y_test)/len(y_test)


# In[8]:


df = pd.read_csv("q1_data.csv")
plt.rcParams["figure.figsize"] = (10,10)
df["pm2.5"].fillna(df["pm2.5"].mean(),inplace=True)

def f(val):
    
    if val=="NW":
        return 1
    if val=="cv":
        return 2
    if val=="NE":
        return 3
    return 4

df["cbwd"] = df["cbwd"].apply(f)
## creating correlation matrix for the dataset
ax = sns.heatmap(df.corr(), annot=True)
df.describe()


# In[9]:


df.isna().sum()


# In[10]:


x = df[["year","day","hour","pm2.5","DEWP","TEMP","PRES","Iws","Is","Ir","cbwd"]]
y = df["month"]


# In[11]:


x_train,x_test,x_val,y_train,y_test,y_val = train_test_val_split(x,y,0.15,0.15)
x_train.shape, x_test.shape, x_val.shape, y_train.shape, y_test.shape, y_val.shape


# In[12]:


## use decision tree
from sklearn.tree import DecisionTreeClassifier

gini_model = DecisionTreeClassifier(criterion="gini")
entropy_model = DecisionTreeClassifier(criterion="entropy")

gini_model.fit(x_train,y_train)
entropy_model.fit(x_train,y_train)

print("gini accuracy on test set:",accuracy(gini_model.predict(x_test),y_test))
print("entropy accuracy on test set:",accuracy(entropy_model.predict(x_test),y_test))


# In[17]:


## since entropy gives better accuracy, we will use entropy as the criterion for subsequent parts

depth = [2,4,8,10,15,30]
train_acc = []
test_acc = []
val_acc = []

for d in depth:
    
    model = DecisionTreeClassifier(criterion="entropy",max_depth=d)
    model.fit(x_train,y_train)
    
    train_acc.append(accuracy(model.predict(x_train),y_train))
    test_acc.append(accuracy(model.predict(x_test),y_test))
    val_acc.append(accuracy(model.predict(x_val),y_val))

plt.rcParams["figure.figsize"] = (10,10)
plt.plot(depth,train_acc,label="train accuracy")
plt.plot(depth,test_acc,label="test accuracy")
plt.plot(depth,val_acc,label="val accuracy")
plt.title("Decision Tree depth vs Accuracy on train and test sets")
plt.xlabel("depth")
plt.ylabel("accuracy")
plt.legend()
plt.show()

print("Train acc:\n",train_acc)
print("Test acc:\n",test_acc)
print("Val acc:\n",val_acc)


# In[11]:


## 1.c

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,criterion="entropy",max_depth=3,max_samples=0.5)
rf.fit(x_train,y_train)

print("train accuracy:",accuracy(rf.predict(x_train),y_train))
print("test accuracy:",accuracy(rf.predict(x_test),y_test))


# In[12]:


## 1.d

depth = [4,8,10,15,20,30]
train_acc = []
test_acc = []
val_acc = []

for d in depth:
    
    model = RandomForestClassifier(n_estimators=100,criterion="entropy",max_depth=d,max_samples=0.5)
    model.fit(x_train,y_train)
    
    train_acc.append(accuracy(model.predict(x_train),y_train))
    test_acc.append(accuracy(model.predict(x_test),y_test))
    val_acc.append(accuracy(model.predict(x_val),y_val))


# In[13]:


plt.rcParams["figure.figsize"] = (10,10)
plt.plot(depth,train_acc,label="train accuracy")
plt.plot(depth,test_acc,label="test accuracy")
plt.plot(depth,val_acc,label="validation accuracy")
plt.title("Decision Tree depth vs Accuracy on train,test,val sets")
plt.xlabel("depth")
plt.ylabel("accuracy")
plt.legend()
plt.show()

print("Train acc:\n",train_acc)
print("Test acc:\n",test_acc)
print("Val acc:\n",val_acc)


# In[14]:


## 1.e

## ask if it's depth or num_estimators?

from sklearn.ensemble import AdaBoostClassifier

depth = [4,8,10,15,20]
train_acc = []
test_acc = []
val_acc = []

for d in depth:
    
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=d),n_estimators=100)
    model.fit(x_train,y_train)
    
    train_acc.append(accuracy(model.predict(x_train),y_train))
    test_acc.append(accuracy(model.predict(x_test),y_test))
    val_acc.append(accuracy(model.predict(x_val),y_val))
    


# In[15]:


plt.rcParams["figure.figsize"] = (10,10)
plt.plot(depth,train_acc,label="train accuracy")
plt.plot(depth,test_acc,label="test accuracy")
plt.plot(depth,val_acc,label="validation accuracy")
plt.title("Decision Tree depth vs Accuracy on train,test,val sets")
plt.xlabel("depth")
plt.ylabel("accuracy")
plt.legend()
plt.show()

print("Train acc:\n",train_acc)
print("Test acc:\n",test_acc)
print("Val acc:\n",val_acc)


# As we observe, for higher values of the depth, AdaBoost outperforms RandomForest model
