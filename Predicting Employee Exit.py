#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from math import inf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv('data/Predicting Employee Exit.csv')
df


# In[7]:


df.drop(['sales', 'salary'], axis = 1, inplace = True)
  
# display
df


# In[4]:


df.head()


# In[8]:


dt = pd.DataFrame(df)
labels = df["left"]
data = df.drop("left",axis=1)


# In[9]:


res=train_test_split(data,labels,train_size=0.8,test_size=0.2,random_state=1)
train_data,test_data,train_labels,test_labels=res
res


# In[10]:


from scipy import stats
import numpy as np
f=open('data/Predicting Employee Exit.csv','r').readlines()
w=f[1].split()
l1=w[1:8]
l2=w[8:15]
list1=[float(x) for x in l1]
list1


# In[12]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(train_data,train_labels)
predicted=knn.predict(test_data)
print("predictions:")
print(predicted)
print("Ground Truth")
print(test_labels)


# In[ ]:




