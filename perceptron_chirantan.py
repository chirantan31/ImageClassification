#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
from random import randint


# In[2]:


def load_dataset(data_dir=''):
    """Load the train and test examples 
    """
    x_train = np.load("data/x_train.npy")
    y_train = np.load("data/y_train.npy")
    x_test = np.load("data/x_test.npy")
    y_test = np.load("data/y_test.npy")

    return x_train, y_train, x_test, y_test


# In[3]:


x_train, y_train, x_test, y_test = load_dataset()


# In[4]:


#x_train, y_train = x_train[:100], y_train[:100]


# In[5]:


classes, counts = np.unique(y_train, return_counts=True)


# In[6]:


W = np.zeros((len(classes), x_train.shape[1]))


# In[7]:


time1 = time.time()
LR = .01
num_epochs = 10
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        x = np.reshape(x, (x.shape[0],1))
        dot = np.dot(W, x)
        #print(dot)
        y_pred = np.argmax(dot)
        if y_pred != y:
            update = LR*x.T
            W[y]+=update[0]
            W[y_pred]-=update[0]
            
        if (y_pred == y):
            total_correct += 1
        
    print("Training Accuracy", total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)


# In[8]:


total_correct = 0
num_samples, num_classified_correctly = np.zeros((len(classes), 1)), np.zeros((len(classes), 1))
avg_accuracy = 0
conf_matrix = np.zeros((len(classes), len(classes)))
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    x = np.reshape(x, (x.shape[0],1))
    dot = np.dot(W, x)
    #print(dot)
    y_pred = np.argmax(dot)
    num_samples[y]+=1
    conf_matrix[y][y_pred]+=1
    if (y_pred == y):
        total_correct += 1
        num_classified_correctly[y] += 1
avg_classification_rate = total_correct/np.float(len(x_test))
avg_class_classification_rate = num_classified_correctly/num_samples
print("Avg Classification Rate", total_correct/np.float(len(x_test) ) )
print("Avg Class Classification Rate", avg_class_classification_rate )
print(conf_matrix)


# In[ ]:




