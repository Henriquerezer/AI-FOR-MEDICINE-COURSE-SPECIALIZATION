#!/usr/bin/env python
# coding: utf-8

# # Count Patients
# 
# Welcome to the first practice lab of this week!

# ## Import Packages

# In[ ]:


import numpy as np
import pandas as pd


# We'll work with data where:
# - Time: days after a disease is diagnosed and the patient either dies or left the hospital's supervision.
# - Event: 
#     - 1 if the patient died
#     - 0 if the patient was not observed to die beyond the given 'Time' (their data is censored)
#     
# Notice that these are the same numbers that you see in the lecture video about estimating survival.

# In[ ]:


df = pd.DataFrame({'Time': [10,8,60,20,12,30,15],
                   'Event': [1,0,1,1,0,1,0]
                  })
df


# ## Count Number of Censored Patients

# In[ ]:


df['Event'] == 0


# Patient 1, 4 and 6 were censored.

# - Count how many patient records were censored
# 
# When we sum a series of booleans, `True` is treated as 1 and `False` is treated as 0.

# In[ ]:


sum(df['Event'] == 0)


# ## Count Number of Patients Who Definitely Survived Past Time $t$
# 
# This assumes that any patient who was censored died at the time of being censored ( **died immediately**).
# 
# If a patient survived past time `t`:
# - Their `Time` of event should be greater than `t`.  
# - Notice that they can have an `Event` of either 1 or 0.  What matters is their `Time` value.

# In[ ]:


t = 25
df['Time'] > t


# In[ ]:


sum(df['Time'] > t)


# ## Count Number of Patients Who May Have Survived Past Time $t$
# 
# This assumes that censored patients **never die**.
# - The patient is censored at any time and we assume that they live forever.
# - The patient died (`Event` is 1) but after time `t`

# In[ ]:


t = 25
(df['Time'] > t) | (df['Event'] == 0)


# In[ ]:


sum( (df['Time'] > t) | (df['Event'] == 0) )


# ## Count Number of Patients Who were Not Censored Before Time $t$

# If patient was not censored before time `t`:
# - They either had an event (death) before `t`, at `t`, or after `t` (any time)
# - Or, their `Time` occurs after time `t` (they may have either died or been censored at a later time after `t`)

# In[ ]:


t = 25
(df['Event'] == 1) | (df['Time'] > t)


# In[ ]:


sum( (df['Event'] == 1) | (df['Time'] > t) )

