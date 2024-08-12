#!/usr/bin/env python
# coding: utf-8

# # Categorical Variables
# 
# Welcome to the first lab of the week!

# ## Import Package

# In[1]:


import pandas as pd
import numpy as np


# ### Which Features are Categorical?

# In[2]:


df = pd.DataFrame({'ascites': [0,1,0,1],
                   'edema': [0.5,0,1,0.5],
                   'stage': [3,4,3,4],
                   'cholesterol': [200.5,180.2,190.5,210.3]
                  })
df


# In this small sample dataset, 'ascites', 'edema', and 'stage' are categorical variables
# - ascites: value is either 0 or 1
# - edema: value is either 0, 0.5 or 1
# - stage: is either 3 or 4
# 
# 'cholesterol' is a continuous variable, since it can be any decimal value greater than zero.

# ### Which Categorical Variables to One-Hot Encode?
# 
# Of the categorical variables, which one should be one-hot encoded (turned into dummy variables)?
# 
# - ascites: is already 0 or 1, so there is not a need to one-hot encode it.
#     - We could one-hot encode ascites, but it is not necessary when there are just two possible values that are 0 or 1.
#     - When values are 0 or 1, 1 means a disease is present, and 0 means normal (no disease).
# - edema: Edema is swelling in any part of the body. This data set's 'edema' feature has 3 categories, so we will want to one-hot encode it so that there is one feature column for each of the three possible values.
#     - 0: No edema
#     - 0.5: Patient has edema, but did not receive diuretic therapy (which is used to treat edema)
#     - 1: Patient has edeam, despite also receiving diuretic therapy (so the condition may be more severe).
# - stage: has values of 3 and 4.  We will want to one-hot encode these because they are not values of 0 or 1.
#     - the "stage" of cancer is either 0, 1,2,3 or 4.  
#     - Stage 0 means there is no cancer.  
#     - Stage 1 is cancer that is limited to a small area of the body, also known as "early stage cancer"
#     - Stage 2 is cancer that has spread to nearby tissues
#     - stage 3 is cancer that has spread to nearby tissues, but more so than stage 2
#     - stage 4 is cancer that has spread to distant parts of the body, also known as "metastatic cancer".
#     - We could convert stage 3 to 0 and stage 4 to 1 for the sake of training a model.  This would may be confusing for anyone reviewing our code and data.  We will one-hot encode the 'stage'.
#         -You'll actually see that we end up with 0 representing stage 3 and 1 representing stage 4 (see the next section).

# ### Multi-collinearity of One-Hot Encoded Features
# 
# Let's see what happens when we one-hot encode the 'stage' feature.
# 
# We'll use [pandas.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)

# In[3]:


df_stage = pd.get_dummies(data=df,
               columns=['stage']
              )
df_stage[['stage_3','stage_4']]


# What do you notice about the 'stage_3' and 'stage_4' features?
# 
# Given that stage 3 and stage 4 are the only possible values for stage,  
# If you know that patient 0 (row 0) has stage_3 set to 1, 
# what can you say about that same patient's value for the stage_4 feature?
# - When stage_3 is 1, then stage_4 must be 0
# - When stage_3 is 0, then stage_4 must be 1
# 
# This means that one of the feature columns is actually redundant.  We should drop one of these features to avoid multicollinearity (where one feature can predict another feature).

# In[4]:


df_stage


# In[5]:


df_stage_drop_first = df_stage.drop(columns='stage_3')
df_stage_drop_first


# Note, there's actually a parameter of pandas.get_dummies() that lets you drop the first one-hot encoded column.  You'll practice doing this in this week's assignment!

# ### Make the Numbers Decimals
# 
# We can cast the one-hot encoded values as floats by setting the data type to numpy.float64.
# - This is helpful if we are feeding data into a model, where the model expects a certain data type (such as a 64-bit float, 32-bit float etc.)

# In[6]:


df_stage = pd.get_dummies(data=df,
               columns=['stage'],
              )
df_stage[['stage_4']]


# In[7]:


df_stage_float64 = pd.get_dummies(data=df,
               columns=['stage'],
               dtype=np.float64
              )
df_stage_float64[['stage_4']]


# In[ ]:




