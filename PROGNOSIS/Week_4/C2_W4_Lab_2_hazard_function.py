#!/usr/bin/env python
# coding: utf-8

# # Hazard Function
# 
# In this week's lab you'll learn about implementing `Hazard Function`.

# ## Import Packages

# In[1]:


import numpy as np
import pandas as pd


# Let's say we fit the hazard function
# $$
# \lambda(t, x) = \lambda_0(t)e^{\theta^T X_i}
# $$
# 
# So that we have the coefficients $\theta$ for the features in $X_i$
# 
# If you have a new patient, let's predict their hazard $\lambda(t,x)$

# In[2]:


lambda_0 = 1
coef = np.array([0.5,2.])
coef


# In[3]:


X = pd.DataFrame({'age': [20,30,40],
                  'cholesterol': [180,220,170]
                 })
X


# - First, let's multiply the coefficients to the features.
# - Check the shapes of the coefficients and the features to decide which one to transpose

# In[4]:


coef.shape


# In[5]:


X.shape


# It looks like the coefficient is a 1D array, so transposing it won't do anything.  
# - We can transpose the X so that we're multiplying a (2,) array by a (2,3) dataframe.
# 
# So the formula looks more like this (transpose $X_i$ instead of $\theta$
# $$
# \lambda(t, x) = \lambda_0(t)e^{\theta X_i^T}
# $$
# 
# - Let's multiply $\theta X_i^T$

# In[6]:


np.dot(coef,X.T)


# Calculate the hazard for the three patients (there are 3 rows in X)

# In[7]:


lambdas = lambda_0 * np.exp(np.dot(coef,X.T))
patients_df = X.copy()
patients_df['hazards'] = lambdas
patients_df


# In[ ]:




