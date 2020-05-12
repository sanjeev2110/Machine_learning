#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import optimize
from ipywidgets import *
from IPython.display import SVG
from sklearn import datasets


# In[2]:


AND = pd.DataFrame({'x1':(0,0,1,1), 'x2':(0,1,0,1), 'y':(0,0,0,1)})
AND  ##  X1 & X2 are features columns and Y is label (AND relation)

# First we need to initialize weights to small,random values(can be positive and negative)
# In[3]:


w=np.random.randn(3)*1e-4

Then, a simple activation function for calculating g(h):
# In[4]:


w


# In[5]:


inputs = AND[['x1','x2']]
target = AND['y']


# In[6]:


g=lambda inputs,weights: np.where(np.dot(inputs, weights)>0,1,0)

Finally, a training function that iterates the learning algorithm, returning the adapted weights
# In[7]:


np.c_[np.array([1,2,3]), np.array([4,5,6])] ##example how np.c works


# In[8]:


inputs=AND[['x1','x2']]
inputs


# In[9]:


-np.ones((len(inputs), 1))


# In[10]:


a=np.c_[inputs, -np.ones((len(inputs), 1))]


# In[11]:


w


# In[12]:


np.dot(a,w)


# In[13]:


def train(inputs, targets, weights, eta, n_iterations):
    # Add the inputs that match the bias node
    inputs = np.c_[inputs, -np.ones((len(inputs),1))]

    for n in range(n_iterations):
        
        activations = g(inputs, weights);
        print(n, activations)
        weights -= eta*np.dot(np.transpose(inputs), activations - targets)
        print(n, weights)
    return(weights)

Let's test it first on the AND function
# In[14]:


inputs = AND[['x1','x2']]
target = AND['y']

w= train(inputs, target, w, 0.001,20)
target

checking the performance
# In[15]:


g(np.c_[inputs, -np.ones((len(inputs), 1))], w)

Thus ,it has learned the fuction perfectly,Now for OR
# In[16]:


OR = pd.DataFrame({'x1':(0,0,1,1), 'x2':(0,1,0,1), 'y':(0,1,1,1)})
OR


# In[18]:


w=np.random.randn(3)*1e-4


# In[22]:


inputs=OR[['x1','x2']]
target=OR['y']

w=train(inputs, target, w, 0.25,20)


# In[23]:


g(np.c_[inputs, -np.ones((len(inputs), 1))], w)


# In[ ]:




