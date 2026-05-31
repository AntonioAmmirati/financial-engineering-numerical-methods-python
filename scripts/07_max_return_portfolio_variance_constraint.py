#!/usr/bin/env python
# coding: utf-8

# (ii) A minimum variance portfolio with 16% expected rate of return can be set up by
# investing a percentage \(w_i\) of the total value of the portfolio in asset \(i\), with \(i = 1:3\),
# where \(w_i\) can be found by solving the following linear system
# 
# Find the weights of each asset in this minimum variance portfolio.

# In[ ]:


import numpy as np


# In[ ]:


P = np.array([[0,0,0,1,0],[0,1,0,0,0],[0,0,1,0,0],[1,0,0,0,0],[0,0,0,0,1]])
b = np.array([0,0,0,1,0.16])
L = np.array([[1,0,0,0,0],[-0.0225,1,0,0,0],[0.0315,0.051852,1,0,0],[0.045,-0.333333,0.038067,1,0],[0.1,0.246914,0.400056,-0.482738,1]])
U = np.array([[1,1,1,0,0],[0,0.2025,0.0645,1,0.15],[0,0,0.210555,0.948148,0.192222],[0,0,0,1.297240,0.142683],[0,0,0,0,-0.045059]])


# In[ ]:


print(L)


# In[ ]:


print(U)


# In[ ]:


#PA = LU
#LUx = Pb
Pb = P @ b #permutation of the b vector


# In[ ]:


print(Pb)


# In[ ]:


#forward substitution
def forward_solver(L,b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i,j]*y[j]
        y[i] = (b[i] - s)/L[i,i]
    return y
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


y = forward_solver(L,Pb)


# In[ ]:


x = backward_solver(U,y)
print(x)


# ### The portfolio weights are the first three components of the solution vector \(x\):
# 
# - \(w1 = 0.23564727\)
# - \(w2 = 0.33011065\)
# - \(w3 = 0.43424208\)
