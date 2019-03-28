#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np


# In[2]:


s = time.time()


# In[3]:


def load_dataset(data_dir=''):
    """Load the train and test examples 
    """
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    x_test = np.load("data/x_test.npy")
    y_test = np.load("data/y_test.npy")

    return x_train, y_train, x_test, y_test


# In[4]:


x_train, y_train, x_test, y_test = load_dataset()


# In[5]:


#x_train, y_train = x_train[:100], y_train[:100]


# In[6]:


classes, counts = np.unique(y_train, return_counts=True)


# In[7]:


def one_hotter(a, colors=256):
    #print(colors)
    b = np.zeros((a.size, colors))
    b[np.arange(a.size),a] = 1
    return b.T


# In[9]:


def train(x_train, y_train):
    start = time.time()
    ans = np.zeros((10,256,784))
    for i, x in enumerate(x_train):
        y = y_train[i]
        hot = one_hotter(x)
        ans[y]+=hot
    print(time.time()-start)
    return ans


# In[10]:


trained = train(x_train, y_train)


# In[11]:


p_class = np.log(counts/sum(counts))
print(p_class)


# In[12]:


def p(pixel, val):
    count = trained[:, val, pixel]
    #total = counts[c]
    return np.log((count + 1)/(counts+1))


# In[13]:


def p_total(x):
    totals = sum([p(pixel, col) for pixel, col in enumerate(x)])
    return totals + p_class


# In[22]:


total_correct = 0
num_samples, num_classified_correctly = np.zeros((len(classes), 1)), np.zeros((len(classes), 1))
avg_accuracy = 0
conf_matrix = np.zeros((len(classes), len(classes)))
most_prototypical_ind, least_prototypical_ind = np.zeros((len(classes))), np.zeros((len(classes)))
most_prototypical_val, least_prototypical_val = np.ones((len(classes)))*-np.inf, np.ones((len(classes)))* np.inf
start = time.time()
correct, all_samples = 0,0
for i, x in enumerate(x_test):
    y = y_test[i]    
    p_tot = p_total(x)
    y_pred = np.argmax(p_tot)
    num_samples[y]+=1
    conf_matrix[y][y_pred]+=1
    
    
    if most_prototypical_val[y_pred] < p_tot[y_pred]:
        most_prototypical_ind[y_pred] = i
        most_prototypical_val[y_pred] = p_tot[y_pred]
    if least_prototypical_val[y_pred] > p_tot[y_pred]:
        least_prototypical_ind[y_pred] = i
        least_prototypical_val[y_pred] = p_tot[y_pred]
    
    if (y_pred == y):
        total_correct += 1
        num_classified_correctly[y] += 1
avg_classification_rate = total_correct/np.float(len(x_test))
avg_class_classification_rate = num_classified_correctly/num_samples
print("Avg Classification Rate", total_correct/np.float(len(x_test) ) )
print("Avg Class Classification Rate", avg_class_classification_rate )
print(conf_matrix)
    
    
print(correct, all_samples)
print(time.time()-start)


# In[23]:


print(most_prototypical_val, most_prototypical_ind)
least_prototypical_val, least_prototypical_ind


# In[14]:


e = time.time()
print(e-s)

