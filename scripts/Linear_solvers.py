#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


#LU DECOMPOSITION
def LU(A):
    U = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n)
    for i in range(n-1):
        pivot = U[i,i]
        if abs(pivot) < 1e-15:
            raise ZeroDivisionError('Pivot zero:pivoting needed')
        for j in range(i+1,n):
            L[j,i] = U[j,i]/pivot
            U[j,i:] -=  L[j,i]*U[i,i:]
    return L,U
            


# In[ ]:


A = np.array([[5,7,3],[2,1,0],[5,0,7]])
LU(A)


# In[ ]:


#SOLVING LINEAR SYSTEMS:
def forward_sub(L,b):
   
    #Ly=b
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i,k] * y[k]
        y[i] = (b[i] -s)/ L[i,i]
    return y
def backward_sub(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        s = 0
        for k in range(i+1,n):
            s += U[i,k] * x[k]
        x[i] = (y[i] - s)/ U[i,i]
    return x


# In[ ]:


#FORWARD SUBSTITUTION WITH BAND-2:
#Lx = b
def forward_solver(L,b):
    n = L.shape[0]
    y = np.zeros(n)
    x[0] = b[0]/L[0,0]
    if n >= 2:
        x[1] = (b[1] -(L[1,0] * x[0]))/L[1,1]
    for i in range(2,n):
        x[i] = ((b[i]-(L[i,i-1] * x[i-1] + L[i,i-2] * x[i-2]))/L[i,i])
    return x 


# In[ ]:


#BACKWARD SUBSTITUTION WITH BAND-2
#Ux=b
def backward_solver(U,b):
    n = U.shape[0]
    x = np.zeros(n)
    x[n-1] = b[n-1]/U[n-1,n-1]
    if n>=2:
        x[n-2] = (b[n-2] -(U[n-2,n-1]*x[n-1]))/U[n-2,n-2]
    for i in range(n-3,-1, -1):
        x[i] = (b[i] - (U[i,i+1]*x[i+1] + U[i,i+2]*x[i+2]))/U[i,i]
    return x
    


# In[ ]:


#EXERCIZES


# In[ ]:


Sigma = np.array([
    [ 1.0,   -0.525,  1.375,  -0.075, -0.75 ],
    [-0.525,  4.0,    0.1875,  0.1875, -0.675],
    [ 1.375,  0.1875, 12.25,   0.4375, -1.875],
    [-0.075,  0.1875, 0.4375,  6.25,    0.3  ],
    [-0.75,  -0.675, -1.875,   0.3,     4.41 ]
], dtype=float)


# In[ ]:


#COVARIANCE OF 5 Random Variables
#find the correlation matrix
# we follow the formula corr(x1,x2) = cov(x1,x2)*sigma(x1)*sigma(x2)
#so corr(x1,x2) = cov(x1,x2)/sigma(x1,x2)
#that in LA this could be written as ( since sigma = sqrt(diagsigma) * C(corr matrix) * (diagsigma)
# C = sqrt(diagsigma)^-1*Sigma * sqrt(diagsigma)^-1


# In[ ]:


Var = np.diag(Sigma)


# In[ ]:


Var


# In[ ]:


std_diag = np.sqrt(Var)


# In[ ]:


std_diag_inv = np.diag(1/std_diag)


# In[ ]:


C = std_diag_inv @ Sigma @ std_diag_inv


# In[ ]:


print(C)


# #### EX.1 HW4

# In[ ]:


#state of market (tau)
a = [ 950 ,1187.50 ,1225, 1300, 1362.50, 1412.50 , 1500 , 1575 , 1700]


# In[ ]:


M = np.array([
    [1175-a[0], 0,        0,        0,        0,        0,        0,        0,        0],  # P1175
    [1200-a[0], 1200-a[1],0,        0,        0,        0,        0,        0,        0],  # P1200
    [1250-a[0], 1250-a[1],1250-a[2],0,        0,        0,        0,        0,        0],  # P1250
    [1350-a[0], 1350-a[1],1350-a[2],1350-a[3],0,        0,        0,        0,        0],  # P1350

    [0,        0,        0,        0,        a[4]-1350,a[5]-1350,a[6]-1350,a[7]-1350,a[8]-1350],  # C1350
    [0,        0,        0,        0,        0,        a[5]-1375,a[6]-1375,a[7]-1375,a[8]-1375],  # C1375
    [0,        0,        0,        0,        0,        0,        a[6]-1450,a[7]-1450,a[8]-1450],  # C1450
    [0,        0,        0,        0,        0,        0,        0,        a[7]-1550,a[8]-1550],  # C1550
    [0,        0,        0,        0,        0,        0,        0,        0,        a[8]-1600]   # C1600
])

print(M.shape)
print(M)


# In[ ]:


# St = MtQ
#so Q = forward + backward solver for M(9,9) since M(4,4) is lower triangular and the successive M(5,5) is upper triangular


# In[ ]:


M_low =  np.array([
    [1175-a[0], 0,        0,        0],  # P1175
    [1200-a[0], 1200-a[1],0,        0],  # P1200
    [1250-a[0], 1250-a[1],1250-a[2],0],  # P1250
    [1350-a[0], 1350-a[1],1350-a[2],1350-a[3]]])# P1350
#St_low prices of put contracts at time t 
St_low = np.array([46.60,51.55,63.30,95.30])
M_upp = np.array([ [ a[4]-1350,a[5]-1350,a[6]-1350,a[7]-1350,a[8]-1350],  # C1350
    [        0,        a[5]-1375,a[6]-1375,a[7]-1375,a[8]-1375],  # C1375
    [        0,        0,        a[6]-1450,a[7]-1450,a[8]-1450],  # C1450
    [       0,        0,        0,        a[7]-1550,a[8]-1550],  # C1550
    [        0,        0,        0,        0,        a[8]-1600]   # C1600
])
#St_up prices of call contracts at time t
St_up = np.array([99.55,84.90,47.25,15.80,7.90])

print(M_low)
print(M_upp)


# In[ ]:


Q_low = forward_sub(M_low,St_low)


# In[ ]:


print(Q_low)


# In[ ]:


M_low


# In[ ]:


Q_up = backward_sub(M_upp,St_up)


# In[ ]:


Q_up


# In[ ]:


det1 = np.linalg.det(M_low)
det2 = np.linalg.det(M_upp)
payoff_matrix_det = det1 * det2


# In[ ]:


print(payoff_matrix_det )


# In[ ]:


Q = np.concatenate((Q_low, Q_up))
St = np.concatenate((St_low,St_up))


# In[ ]:


Q


# In[ ]:


St_test = M @ Q


# In[ ]:


St_test


# In[ ]:


from sklearn.metrics import root_mean_squared_error
RMSE = root_mean_squared_error(St,St_test)
print(f'RMSE:{RMSE}')


# In[ ]:


def forward_solver(L,b):
    n = L.shape[0]
    y =np.zeros(n)
    
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i,k]*y[k]
        y[i] = (b[i] - (s ))/L[i,i]
    return y


# In[ ]:


q_l = forward_solver(M_low,St_low)


# In[ ]:


q_l


# In[ ]:


def backward_solv(U,y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        s = 0.0
        for k in range(i+1):
            s += U[i,k]*x[k]
        x[i] = (b[i] - s)/U[i,i]
    return x

