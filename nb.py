#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np


# In[5]:


def load_dataset(data_dir=''):
    """Load the train and test examples 
    """
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    x_test = np.load("data/x_test.npy")
    y_test = np.load("data/y_test.npy")

    return x_train, y_train, x_test, y_test


# In[11]:


x_train, y_train, x_test, y_test = load_dataset()


# In[12]:


#x_train, y_train = x_train[:100], y_train[:100]


# In[16]:


classes, counts = np.unique(y_train, return_counts=True)


# In[13]:


def one_hotter(a, colors=256):
    #print(colors)
    b = np.zeros((a.size, colors))
    b[np.arange(a.size),a] = 1
    return b.T


# In[14]:


def train(x_train, y_train):
    start = time.time()
    ans = np.zeros((10,256,784))
    for i, x in enumerate(x_train):
        y = y_train[i]
        hot = one_hotter(x)
        ans[y]+=hot
    print(time.time()-start)
    return ans


# In[15]:


trained = train(x_train, y_train)


# In[17]:


p_class = np.log(counts/sum(counts))
print(p_class)


# In[51]:


def p(pixel, val):
    count = trained[:, val, pixel]
    #total = counts[c]
    return np.log((count + 1)/(counts+1))


# In[86]:


def p_total(x):
    #print(c, x.shape)
    #total = sum([p(pixel, col) for pixel, col in enumerate(x)])
    totals = sum([p(pixel, col) for pixel, col in enumerate(x)])
    return totals + p_class


# In[90]:


start = time.time()
correct, all_samples = 0,0
for i, x in enumerate(x_test):
    p_t = p_total(x)
    #print(p_t)
    if np.argmax(p_t) == y_test[i]:
        correct+=1
    all_samples+=1
print(correct, all_samples)
print(time.time()-start)

