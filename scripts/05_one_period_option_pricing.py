#!/usr/bin/env python
# coding: utf-8

# ##### HW4 - EX(1)
# This question refers to the March 9, 2012 S&P options price data from Table 3.1 in section
# 3.5 of the NLA Primer.
# Consider a one period market model with the following nine securities:
# P1175; P1200; P1250; P1350; C1350; C1375; C1450; C1550; C1600.
# Seven of the states of the market correspond to the midpoints between the strikes of the
# options above, i.e.,
# ω2 : {S(τ ) = 1187.50}; ω3 : {S(τ ) = 1225};
# ω4 : {S(τ ) = 1300}; ω5 : {S(τ ) = 1362.50};
# ω6 : {S(τ ) = 1412.50}; ω7 : {S(τ ) = 1500};
# ω8 : {S(τ ) = 1575};
# The first and last state are any possible combination of the first state being that the value
# of the index at the options maturity is 800, 950, 1100, and the last state being that the value of
# the index at the options maturity is 1650, 1700, 1800. (In other words, there are nine diﬀerent
# one period market models to consider.)
# For each of the nine one period market models, find the payoﬀ matrix Mτ , find the
# state prices vector Q and determine whether the model is arbitrage–free, and if the model
# is arbitrage–free, compute the RMS error of this model.

# ##### From Table (3.1) we find the vector St0 , corresponding to the prices of the selected option contracts.In the end we will try all the 9 combinationation with a for loop.

# In[ ]:


import numpy as np
St_low = np.array([46.60,51.55,63.30,95.30])
St_up = np.array([99.55,84.90,47.25,15.80,7.90])
#We start to write it separated , to match our future forward and backward solver to find Q


# In[ ]:


#Payoff matrix:
#state of the market (ω),I choose for this test ω1 = 950 and ω9 = 1800; in the end we will test all the possible combination.
a = [ 950,1187.50 ,1225, 1300, 1362.50, 1412.50 , 1500 , 1575 , 1800]


# In[ ]:


#payoff matrix
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
#so Q = forward + backward solver for M(9,9) since M(4,4) is lower triangular and the successive M(5,5) is upper triangular.


# In[ ]:


#let's start dividing the matrix M in two matrices.
M_low =  np.array([
    [1175-a[0], 0,        0,        0],  # P1175
    [1200-a[0], 1200-a[1],0,        0],  # P1200
    [1250-a[0], 1250-a[1],1250-a[2],0],  # P1250
    [1350-a[0], 1350-a[1],1350-a[2],1350-a[3]]])# P1350

M_upp = np.array([ [ a[4]-1350,a[5]-1350,a[6]-1350,a[7]-1350,a[8]-1350],  # C1350
    [        0,        a[5]-1375,a[6]-1375,a[7]-1375,a[8]-1375],  # C1375
    [        0,        0,        a[6]-1450,a[7]-1450,a[8]-1450],  # C1450
    [       0,        0,        0,        a[7]-1550,a[8]-1550],  # C1550
    [        0,        0,        0,        0,        a[8]-1600]   # C1600
])

print(M_low)
print(M_upp)


# In[ ]:


#now we prepare the functions for forward/backward substitution.
def forward_solver(L,b):
    n = L.shape[0]
    x = np.zeros(n,dtype = float)
    for i in range(n):
        s = 0.0 
        for k in range(i):     #k start before i , ex. (L[2,k] * b[k]) k = 1
            s += L[i,k] * x[k] #we need an accumulative variable to make the loop works 
        x[i] = (b[i] - s)/L[i,i]  # ex. x[3] = (b[3] - (L[3,1]*x[1] + L[3,2] * x[2]))/L[3,3]
    return x
def backward_solver(U,b):
    n = U.shape[0]
    x = np.zeros(n, dtype = float)
    for i in range(n-1,-1,-1):
        s = 0.0 
        for k in range(i+1,n ):  #k runs over entries after i
            s += U[i,k]*x[k]
        x[i] = (b[i] -s)/U[i,i]
    return x


# In[ ]:


Q_low = forward_solver(M_low,St_low)
Q_up = backward_solver(M_upp , St_up)


# In[ ]:


print(Q_low)
print(Q_up)


# In[ ]:


#let's concatenate
Q = np.concatenate((Q_low,Q_up))
St0 = np.concatenate((St_low , St_up))


# In[ ]:


print(Q)
#Componenents of the vector Q are <= 0 , so the model is not arbitrage free


# In[ ]:


#following the formula = St0 = MtQ
St_test = M @ Q
print(St_test)


# In[ ]:


#let's check with all prices -St0
w1_list = [800,950,1100]
w9_list = [1650,1700,1800]
for w1 in w1_list:
    for w9 in w9_list:
        a[0] = w1
        a[8] = w9
        M_low =  np.array([
    [1175-a[0], 0,        0,        0],  # P1175
    [1200-a[0], 1200-a[1],0,        0],  # P1200
    [1250-a[0], 1250-a[1],1250-a[2],0],  # P1250
    [1350-a[0], 1350-a[1],1350-a[2],1350-a[3]]])# P1350
        M_upp = np.array([ [ a[4]-1350,a[5]-1350,a[6]-1350,a[7]-1350,a[8]-1350],  # C1350
    [        0,        a[5]-1375,a[6]-1375,a[7]-1375,a[8]-1375],  # C1375
    [        0,        0,        a[6]-1450,a[7]-1450,a[8]-1450],  # C1450
    [       0,        0,        0,        a[7]-1550,a[8]-1550],  # C1550
    [        0,        0,        0,        0,        a[8]-1600]   # C1600
])
    
        Q_low = forward_solver(M_low,St_low)
        Q_up = backward_solver(M_upp , St_up)
        Q = np.concatenate((Q_low,Q_up))
        if np.all(Q>0):
           print(f'model arbitrage free with {w1} and {w9}')
        else:
           print(f'with {w1} and {w9} the model is not Arbitrage Free, since Q(i)<=0 ')


# #### RESULTS (SUMMARY)
# For all nine one-period market models (generated by $w_1 \in \{800,950,1100\}$ and $w_9 \in \{1650,1700,1800\}$), the state-price vector $Q$ was obtained by solving $M_\tau Q = S_0$.
# 
# In every case, at least one component of $Q$ is non-positive ($Q_i \le 0$). Therefore, none of the nine models is arbitrage-free.
# 
# Hence, no arbitrage-free model is available among the nine candidates, and the RMS error is not applicable for an arbitrage-free fit.
