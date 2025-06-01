#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[19]:


np.__version__


# In[20]:


pip list


# In[21]:


breast_cancer = sklearn.datasets.load_breast_cancer()


# In[22]:


breast_cancer


# In[23]:


breast_cancer.data


# In[24]:


breast_cancer.target


# In[25]:


X = breast_cancer.data


# In[26]:


Y = breast_cancer.target


# In[27]:


breast_cancer.feature_names


# In[28]:


breast_cancer.keys()


# In[29]:


data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)


# In[30]:


data


# In[31]:


breast_cancer.keys()


# In[32]:


data["Class"] = breast_cancer.target


# In[33]:


data


# In[34]:


breast_cancer.target_names


# In[36]:


data['Class'].value_counts()


# In[39]:


data.groupby("Class").mean()

1 - benign
0 - malingnant
# In[ ]:


Y.mean() , Y_test.mean(), Y_train.mean()


# In[63]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1,stratify=Y, random_state=1)


# In[64]:


Y.mean() , Y_test.mean(), Y_train.mean()


# In[72]:


classifier = LogisticRegression()


# In[73]:


classifier.fit(X_train,Y_train)

TRAINING DATA
# In[76]:


prediction_on_training_data = classifier.predict(X_train)


# In[79]:


accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[82]:


print("Accuracy on trainig data: ",accuracy_on_training_data)

TEST DATA
# In[83]:


prediction_on_test_data = classifier.predict(X_test)


# In[84]:


accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[85]:


print("Accuracy on test data: ",accuracy_on_test_data)


# In[142]:


user_input = input("Enter 30 comma separated values: ")


# In[152]:


print("""
        Each value corresponds to this 30 features: 
      ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error',
       'fractal dimension error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'worst concave points',
       'worst symmetry', 'worst fractal dimension'] """)


# In[143]:


input_data = [float(x) for x in user_input.split(',')]


# In[144]:


input_data_as_numpy_array = np.asarray(input_data)


# In[145]:


input_data_reshape = input_data_as_numpy_array.reshape(1,-1)


# In[146]:


prediction = classifier.predict(input_data_reshape)


# In[147]:


print(prediction)


# In[148]:


if prediction==0:
    print("Malingnant (Cancerous)")
    
else:
    print("Benign (Non-cancerous)")


# In[149]:


breast_cancer.feature_names


# In[ ]:




