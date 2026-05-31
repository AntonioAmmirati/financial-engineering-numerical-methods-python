#!/usr/bin/env python
# coding: utf-8

# #### HW3 - ex.3 (iii)

# In[ ]:


import numpy as np


# In[ ]:


A = np.array([[1.5,101.5,0,0],[2,2,102,0],[0,6,0,106],[2.5,2.5,2.5,102.5]])
L = np.array([[1,0,0,0],[0.6,1,0,0],[0.8,0,1,0],[0,0.06,0.0009,1]])
U = np.array([[2.5,2.5,2.5,102.5],[0,100,-1.5,-61.5],[0,0,100,-82],[0,0,0,109.7638]])
b = np.array([101.30,102.95,107.35,105.45])
Pb = np.array([b[3],b[0],b[1],b[2]]) # Permutation(4,1,2,3)


# In[ ]:


#forward substitution
def forward_solver(L,b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i,j] * y[j]
        y[i] = (b[i] - s)/L[i,i]
    return y


# In[ ]:


y = forward_solver(L,Pb)


# In[ ]:


#backward substitution
def backward_solver(U,y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        s = 0.0
        for j in range(i+1,n):
            s += U[i,j]*x[j]
        x[i] = (y[i] - s)/U[i,i]
    return x


# In[ ]:


x = backward_solver(U,y)


# In[ ]:


print(x)


# In[ ]:


#Check 
np.allclose((A @ x),b)


# ##### Results (iii) / Discount Factor = D
# D(4 months) = 0.986040
# D(10 months) = 0.983457
# D(16 months) = 0.970696
# D(22 months) = 0.957068
