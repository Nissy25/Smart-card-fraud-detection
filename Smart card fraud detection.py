#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[8]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('creditcard.csv')


# In[9]:


# first 5 rows of the dataset
credit_card_data.head()


# In[10]:


credit_card_data.tail()


# In[11]:


# dataset informations
credit_card_data.info()


# In[12]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[13]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[14]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[15]:


print(legit.shape)
print(fraud.shape)


# In[16]:


# statistical measures of the data
legit.Amount.describe()


# In[17]:


fraud.Amount.describe()


# In[18]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# In[19]:


legit_sample = legit.sample(n=492)


# In[20]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[21]:


new_dataset.head()


# In[22]:


new_dataset.tail()


# In[23]:


new_dataset['Class'].value_counts()


# In[24]:


new_dataset.groupby('Class').mean()


# In[25]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[26]:


print(X)


# In[27]:


print(Y)


# In[28]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[29]:


print(X.shape, X_train.shape, X_test.shape)


# In[30]:


model = LogisticRegression()


# In[31]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[32]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[33]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[34]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[35]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[ ]:





# In[ ]:




