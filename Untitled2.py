#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[4]:


#Load Dataset
data=pd.read_csv('Raisin_Dataset (1).csv')
data.head()


# In[5]:


#Key Statistics
data.describe()


# In[10]:


# Visualization of Correlations
fig = plt.figure(figsize=(15, 5))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="Purples")
plt.show()


# In[32]:


# Pairplot with hue=Outcome
sns.pairplot(data, hue ='Class')


# In[31]:


#Create x and y variables
x=data.drop('Class', axis=1).to_numpy()
y=data['Class'].to_numpy()

#Create Training and Test Datasets
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x, y, stratify=y,test_size=0.2,random_state=100)

#Scale the Data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train2 = sc.fit_transform(x_train)
x_test2 = sc.transform(x_test)


# In[33]:


#Script for SVM and NB
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix  

for name,method in [('SVM', SVC(random_state=100)),
                    ('Naive Bayes',GaussianNB())]: 
    method.fit(x_train2,y_train)
    predict = method.predict(x_test2)
    print('\nEstimator: {}'.format(name)) 
    print(confusion_matrix(y_test,predict))  
    print(classification_report(y_test,predict)) 


# In[ ]:




