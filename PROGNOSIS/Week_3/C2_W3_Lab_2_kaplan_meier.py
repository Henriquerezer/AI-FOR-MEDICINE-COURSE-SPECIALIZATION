#!/usr/bin/env python
# coding: utf-8

# 
# # Kaplan-Meier
# 
# The Kaplan Meier estimate of survival probability is:
# 
# $$
# S(t) = \prod_{t_i \leq t} (1 - \frac{d_i}{n_i})
# $$
# 
# - $t_i$ are the events observed in the dataset 
# - $d_i$ is the number of deaths at time $t_i$
# - $n_i$ is the number of people who we know have survived up to time $t_i$.
# 
# ## Import Packages

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.DataFrame({'Time': [3,3,2,2],
                   'Event': [0,1,0,1]
                  })
df


# ## Find Those Who Survived Up to Time $t_i$
# 
# If they survived up to time $t_i$, 
# - Their `Time` is either greater than $t_i$
# - Or, their `Time` can be equal to $t_i$

# In[3]:


t_i = 2
df['Time'] >= t_i


# You can use this to help you calculate $n_i$

# ## Find Those Who Died at Time $t_i$
# 
# - If they died at $t_i$:
# - Their `Event` value is 1.  
# - Also, their `Time` should be equal to $t_i$

# In[4]:


t_i = 2
(df['Event'] == 1) & (df['Time'] == t_i)


# You can use this to help you calculate $d_i$
# 
# You'll implement Kaplan Meier in this week's assignment!
