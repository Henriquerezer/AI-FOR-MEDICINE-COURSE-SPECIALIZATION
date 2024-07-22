#!/usr/bin/env python
# coding: utf-8

# # Imputation

# We will use imputation functions provided by scikit-learn.  See the scikit-learn [documentation on imputation](https://scikit-learn.org/stable/modules/impute.html#iterative-imputer).

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.DataFrame({"feature_1": [0,1,2,3,4,5,6,7,8,9,10],
                   "feature_2": [0,np.NaN,20,30,40,50,60,70,80,np.NaN,100],
                  })
df


# ### Mean Imputation

# In[3]:


from sklearn.impute import SimpleImputer


# In[4]:


mean_imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
mean_imputer


# In[5]:


mean_imputer.fit(df)


# In[6]:


nparray_imputed_mean = mean_imputer.transform(df)
nparray_imputed_mean


# Notice how the missing values are replaced with `50` in both cases.

# ### Regression Imputation

# In[7]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[8]:


reg_imputer = IterativeImputer()
reg_imputer


# In[9]:


reg_imputer.fit(df)


# In[10]:


nparray_imputed_reg = reg_imputer.transform(df)
nparray_imputed_reg


# Notice how the filled in values are replaced with `10` and `90` when using regression imputation. The imputation assumed a linear relationship between feature 1 and feature 2.

# In[ ]:




