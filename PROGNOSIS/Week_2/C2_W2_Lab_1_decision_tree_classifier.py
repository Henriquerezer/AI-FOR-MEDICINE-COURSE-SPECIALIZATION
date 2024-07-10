#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classifier
# 
# In this notebook, you'll learn about building a decision tree classifier.

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


X = pd.DataFrame({"feature_1":[0,1,2,3]})
y = pd.Series([0,0,1,1])


# In[ ]:


X


# In[ ]:


y


# In[ ]:


dt = DecisionTreeClassifier()
dt


# In[ ]:


dt.fit(X,y)


# ### Set tree parameters

# In[ ]:


dt = DecisionTreeClassifier(criterion='entropy',
                            max_depth=10,
                            min_samples_split=2
                           )
dt


# ### Set parameters using a dictionary
# 
# - In Python, we can use a dictionary to set parameters of a function.
# - We can define the name of the parameter as the 'key', and the value of that parameter as the 'value' for each key-value pair of the dictionary.

# In[ ]:


tree_parameters = {'criterion': 'entropy',
                   'max_depth': 10,
                   'min_samples_split': 2
                  }


# - We can pass in the dictionary and use `**` to 'unpack' that dictionary's key-value pairs as parameter values for the function.

# In[ ]:


dt = DecisionTreeClassifier(**tree_parameters)
dt

