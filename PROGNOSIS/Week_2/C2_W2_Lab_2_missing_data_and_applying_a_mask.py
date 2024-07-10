#!/usr/bin/env python
# coding: utf-8

# # Missing Data and Applying a Mask

# ## Missing Values

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.DataFrame({"feature_1": [0.1,np.NaN,np.NaN,0.4],
                   "feature_2": [1.1,2.2,np.NaN,np.NaN]
                  })
df


# ### Check if each value is missing

# In[3]:


df.isnull()


# ### Check if any values in a row are true
# 

# In[4]:


df_booleans = pd.DataFrame({"col_1": [True,True,False],
                            "col_2": [True,False,False]
                           })
df_booleans


# - If we use pandas.DataFrame.any(), it checks if at least one value in a column is `True`, and if so, returns `True`.
# - If all rows are `False`, then it returns `False` for that column

# In[5]:


df_booleans.any()


# - Setting the axis to zero also checks if any item in a column is `True`

# In[6]:


df_booleans.any(axis=0)


# - Setting the axis to `1` checks if any item in a **row** is `True`, and if so, returns true
# - Similarily only when all values in a row are `False`, the function returns `False`.

# In[7]:


df_booleans.any(axis=1)


# ### Sum booleans

# In[8]:


series_booleans = pd.Series([True,True,False])
series_booleans


# - When applying `sum` to a series (or list) of booleans, the `sum` function treats `True` as 1 and `False` as zero.

# In[9]:


sum(series_booleans)


# You will make use of these functions in this week's assignment!

# ## Apply a Mask
# 
# Use a 'mask' to filter data of a dataframe

# In[11]:


import pandas as pd


# In[12]:


df = pd.DataFrame({"feature_1": [0,1,2,3,4]})
df


# In[13]:


mask = df["feature_1"] >= 3
mask


# In[14]:


df[mask]


# ### Combining comparison operators
# 
# You'll want to be careful when combining more than one comparison operator, to avoid errors.
# - Using the `and` operator on a series will result in a `ValueError`, because it's 

# In[15]:


df["feature_1"] >=2


# In[16]:


df["feature_1" ] <=3


# In[17]:


# NOTE: This will result in a ValueError
df["feature_1"] >=2 and df["feature_1" ] <=3


# ### How to combine two logical operators for Series
# What we want is to look at the same row of each of the two series, and compare each pair of items, one row at a time. To do this, use:
# - the `&` operator instead of `and`
# - the `|` operator instead of `or`.
# - Also, you'll need to surround each comparison with parenthese `(...)`

# In[18]:


# This will compare the series, one row at a time
(df["feature_1"] >=2) & (df["feature_1" ] <=3)


# In[ ]:




