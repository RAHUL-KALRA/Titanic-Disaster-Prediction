#!/usr/bin/env python
# coding: utf-8

# # Importing the Data

# In[82]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[83]:


data_t = pd.read_csv("C:\\train.csv")


# In[84]:


data_t.head()


# In[85]:


data_t.describe()


# # Finding Out Null Values

# In[86]:


data_t.isna()


# # The number of Male and Female passengers Survived or died

# In[87]:


sns.countplot(x = 'Survived', data=data_t, hue='Sex')


# In[ ]:





# # Null Values recoganition

# In[88]:


data_t.isna().sum()


# In[89]:


sns.heatmap(data_t.isna())


# # Finding out the percentage of null values

# In[90]:


((data_t['Age'].isna().sum())/len(data_t))*100 #can be imputed


# In[91]:


((data_t['Cabin'].isna().sum())/len(data_t))*100 #should be discarded


# # age demographic
# 

# In[92]:


sns.displot(x='Age', data=data_t)


# # Imputing Age and Null Values

# In[93]:


data_t['Age'].mean()


# In[94]:


data_t['Age'].fillna(data_t['Age'].mean(), inplace=True)


# In[95]:


data_t['Age'].isna().sum()


# In[96]:


#now null values are 0 in the age array of the data set
#as shown in the heatmap below
sns.heatmap(data_t.isna())


# # From here we need to drop the Cabin Colum as it has no quantitative significane

# In[97]:


data_t.drop('Cabin', axis=1, inplace=True)


# In[98]:


#check the contents of the data


# In[99]:


data_t.head()


# # Segragating the numerical and object value

# In[100]:


data_t.dtypes


# # Name, Sex and Embarked are object type

# # Name and embarked are not of use, so we will get dummies for Sex

# In[101]:


pd.get_dummies(data_t['Sex'])


# # We only need One column for predection

# In[102]:


#so, we will drop one of the columns
pd.get_dummies(data_t['Sex'], drop_first=True)


# # Adding the above column in the dataset

# In[103]:


gen = pd.get_dummies(data_t['Sex'], drop_first=True)


# In[104]:


data_t['Gender']=gen


# In[105]:


data_t.head()


# # Dropping the unwanted columns

# In[106]:


data_t.drop(['Name','Sex','Embarked','Ticket'], axis=1, inplace=True)


# In[108]:


#Confermation of the drop


# In[109]:


data_t.head()


# In[110]:


#now we only have numerical or quantitative values


# # Segragation of dependent and indepenmdent variables

# In[121]:


#x is independent and y is dependent
x=data_t[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Gender']]
y=data_t['Survived']


# In[122]:


x


# In[123]:


y


# In[124]:


#IMPORTING THE TRAIN TEST SPLIT METHOD


# In[130]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[132]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)


# In[133]:


lor = LogisticRegression()


# In[135]:


lor.fit(x_train,y_train)


# In[137]:


pred = lor.predict(x_test)


# # Confusion Matrix

# In[139]:


from sklearn.metrics import confusion_matrix


# In[142]:


pd.DataFrame(confusion_matrix(y_test, pred), columns=['True Negative', 'True positive'], index=['Actual Negative', 'Actual Positive'])


# # Checking the Accuracy of the Model

# In[143]:


from sklearn.metrics import classification_report


# In[148]:


print(classification_report(y_test, pred))


# In[ ]:




