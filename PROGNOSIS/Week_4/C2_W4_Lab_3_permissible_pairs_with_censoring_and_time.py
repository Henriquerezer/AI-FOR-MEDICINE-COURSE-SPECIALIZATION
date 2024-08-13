#!/usr/bin/env python
# coding: utf-8

# # Permissible Pairs with Censoring and Time
# 
# Welcome to the last practice lab of this week and of this course!

# ## Import Package

# In[1]:


import pandas as pd


# In[2]:


df = pd.DataFrame({'time': [2,4,2,4,2,4,2,4],
                   'event': [1,1,1,1,0,1,1,0],
                   'risk_score': [20,40,40,20,20,40,40,20] 
                  })
df


# We made this data sample so that you can compare pairs of patients visually.

# ### When At Least One Patient is Not Censored
# - A pair may be permissible if at least one patient is not censored.
# - If both pairs of patients are censored, then they are definitely not a permissible pair.

# In[3]:


pd.concat([df.iloc[0:1],df.iloc[1:2]],axis=0)


# In[4]:


if df['event'][0] == 1 or df['event'][1] == 1:
    print(f"May be a permissible pair: 0 and 1")
else:
    print(f"Definitely not permissible pair: 0 and 1")


# In[5]:


pd.concat([df.iloc[4:5],df.iloc[7:8]],axis=0)


# In[6]:


if df['event'][4] == 1 or df['event'][7] == 1:
    print(f"May be a permissible pair: 4 and 7")
else:
    print(f"Definitely not permissible pair: 4 and 7")


# ### If Neither Patient was Censored:
# - If both patients had an event (neither one was censored). This is definitely a permissible pair.

# In[7]:


pd.concat([df.iloc[0:1],df.iloc[1:2]],axis=0)


# In[8]:


if df['event'][0] == 1 and df['event'][1] == 1:
    print(f"Definitely a permissible pair: 0 and 1")
else:
    print(f"May be a permissible pair: 0 and 1")


# ### When One Patient is Censored:
# - If we know that one patient was censored and one had an event, then we can check if censored patient's time is at least as great as the uncensored patient's time.  If so, it's a permissible pair as well

# In[9]:


pd.concat([df.iloc[6:7],df.iloc[7:8]],axis=0)


# In[10]:


if df['time'][7] >= df['time'][6]:
    print(f"Permissible pair: Censored patient 7 lasted at least as long as uncensored patient 6")
else:
    print("Not a permisible pair")


# In[11]:


pd.concat([df.iloc[4:5],df.iloc[5:6]],axis=0)


# In[12]:


if df['time'][4] >= df['time'][5]:
    print(f"Permissible pair")
else:
    print("Not a permisible pair: patient 4 was censored before patient 5 had their event")


# In[ ]:




