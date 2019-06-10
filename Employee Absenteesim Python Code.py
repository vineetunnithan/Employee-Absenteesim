#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Make sure to install all these libraries
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics


# In[2]:


Absent_work = pd.read_excel("https://s3-ap-southeast-1.amazonaws.com/edwisor-india-bucket/projects/data/DataN0101/Absenteeism_at_work_Project.xls")


# In[3]:


Absent_work


# In[4]:


Absent_work.describe() # We can see that all the attributes are treated as integers which is not desired


# In[5]:


# Further checking the data type of the attributes
Absent_work.info(verbose = True)


# In[6]:


# Cleaning the dataset
Absent_work.columns


# In[7]:

#Dropping 'ID' column since it is not important
Absent_work.drop(['ID'], axis = 1,inplace = True)
Absent_work.columns


# In[8]:


#### Preprocessing the dataset#######
#Changing attributes to factors as they are taken as integer in python
# Checking NA values. And removing the missing value rows
Absent_work.isnull().sum()


# In[9]:


# Dropping the missing values
Absent_work.dropna(inplace=True)
Absent_work.isnull().sum()
data_svm = Absent_work


# In[10]:


# Changing the attribtues to categorical
Absent_work['Reason for absence']  = Absent_work['Reason for absence'].astype('category')
Absent_work['Seasons']  = Absent_work['Seasons'].astype('category')
Absent_work['Day of the week']  = Absent_work['Day of the week'].astype('category')
Absent_work['Disciplinary failure']  = Absent_work['Disciplinary failure'].astype('category')
Absent_work['Social drinker']  = Absent_work['Social drinker'].astype('category')
Absent_work['Social smoker']  = Absent_work['Social smoker'].astype('category')

Absent_work.dtypes


# In[11]:


get_ipython().magic('matplotlib inline')
Absent_work.hist(column=None, by=None, grid=True, xlabelsize=None, xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False, figsize=(20,20), layout=(5,3), bins=10)


# In[12]:


get_ipython().magic('matplotlib inline')
boxplot = Absent_work.boxplot(column=['Month of absence', 
       'Transportation expense', 'Distance from Residence to Work',
       'Service time', 'Age', 'Work load Average/day ', 'Hit target', 'Education', 'Son', 'Pet', 'Weight', 'Height', 'Body mass index',
       'Absenteeism time in hours'], return_type='axes', figsize = (30,30))
boxplot


# In[56]:


##### Training the model#####
# Making the train and test split
from sklearn.model_selection import train_test_split
train, test = train_test_split(Absent_work, test_size=0.2)
train_svm, test_svm = train_test_split(data_svm, test_size=0.2)

# In[57]:


print("Length of train set: "+str(len(train)))
print("Length of test set: " +str(len(test)))


# In[88]:


# Training set creation
train_x = train.drop(['Absenteeism time in hours'], axis=1)
train_y = train['Absenteeism time in hours']

# Test set creation

test_x = test.drop(['Absenteeism time in hours'], axis=1)
test_y = test['Absenteeism time in hours']


# In[89]:


print("column name of train dataset:  ", train_x.columns, "\n \n column name of the test dataset: " ,test_x.columns)


# In[90]:


from sklearn.tree import DecisionTreeRegressor 


# In[91]:


d_tree = DecisionTreeRegressor(random_state = 0)


# In[92]:


model = d_tree.fit(train_x, train_y)


# In[95]:


#Prediction
predict_value = model.predict(test_x)
predict_value


# In[110]:


# Calculate RMSE for the predicted values
print((((predict_value - test_y)**2).mean())**(1/2))


# In[ ]:
# SVM Regression
# Training set creation
train_svm_x = train_svm.drop(['Absenteeism time in hours'], axis=1)
train_svm_y = train_svm['Absenteeism time in hours']
train_svm_y = train_svm_y.astype('int')

## Test set creation

test_svm_x = test_svm.drop(['Absenteeism time in hours'], axis=1)
test_svm_y = test_svm['Absenteeism time in hours']

#from sklearn.svm import SVR

svm_model = SVR(gamma='scale', C=1.0, epsilon=0.1)
model_svm = svm_model.fit(train_svm_x, train_svm_y)
train_svm_y.info()




# In[ ]:




