#!/usr/bin/env python
# coding: utf-8

# In[ ]:


x_train = np.array([[1,0,0,1,1,1],
                    [1,1,0,1,1,1],
                    [0,1,0,0,0,0],
                    [0,0,1,0,0,1],
                    [1,1,0,1,1,0],
                    [0,1,0,1,1,0]              
                   ])
print(x_train)
yes= x_train[:3:]
print(yes)
yes_prob1=yes.sum(axis=0)/yes.shape[0]
print(yes_prob1)
no= x_train[3::]
print(no)
no_prob1=no.sum(axis=0)/no.shape[0]
print(no_prob1)
no_prob0=1-no_prob1
print(no_prob0)


# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[111]:


#loading data
trainX=np.loadtxt(fname='trainX.txt',dtype=np.uint8)
testX=np.loadtxt(fname='testX.txt',dtype=np.uint8) 
testY=np.loadtxt(fname='testY.txt',dtype=np.uint8)
#two's and fours'
two=trainX[:250:]
four=trainX[250::]
#calculating probabilities
two_prob1=((two.sum(axis=0))+1)/(two.shape[0]+2)
two_prob0=1-two_prob1
four_prob1=((four.sum(axis=0))+1)/(four.shape[0]+2)
four_prob0=1-four_prob1
p_two=p_four=250/500;
#predicting answers
check=[]
for row in testX:
        check.append(2 if (np.prod(np.where(row[::]==0,p_two*two_prob0,p_two*two_prob1))>np.prod(np.where(row[::]==0,p_four*four_prob0,p_four*four_prob1))) else 4)
print("Predicted answer:",check)
check=np.array(check)
#calculation accuracy
Tp=Tn=Fp=Fn=0
for i in range(check.shape[0]):
    if check[i]==2 and testY[i]==2:
        Tp=Tp+1;
    if check[i]==4 and testY[i]==4:
        Tn=Tn+1;  
    if check[i]==2 and testY[i]==4:
        Fp=Fp+1;
    if check[i]==4 and testY[i]==2:
        Fn=Fn+1;

accuracy=((Tp+Tn)/(Tp+Fp+Tn+Fn))*100
print('Accuracy:',accuracy)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




