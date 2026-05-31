#!/usr/bin/env python
# coding: utf-8

# ### Hw2 - EX (5)
# i) Write the pseudocode for the forward
# substitution corresponding to a
# lower triangular banded matrix of band 2, i.e., for solving Ly= b where b is an n × 1 vector and
# L is an n × n lower triangular matrix such that
# L(j, k) = 0, ∀ 1 ≤ j, k ≤ n with j− k > 2.
# The input for the pseudocode are the matrix L and the vector b; the output is the
# vector y.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


#FORWARD SUBSTITUTION - LOWER TRIANGULAR BAND 2
def forward_solver(L,b):
    n = L.shape[0]
    y = np.zeros(n)
    y[0] = b[0]/L[0,0]
    if n >= 2:
        y[1] = (b[1] -(L[1,0] * y[0]))/L[1,1]
    for i in range(2,n):
        y[i] = ((b[i]-(L[i,i-1] * y[i-1] + L[i,i-2] * y[i-2]))/L[i,i])
    return y 


# In[ ]:


#TEST
L = np.array([[2,0,0,0],[2,1,0,0],[3,1,2,0],[0,2,3,1]])
b = np.array([1,5,7,3])
print(L)


# In[ ]:


y = forward_solver(L,b)
print(y)
print(L @ y , b) # Ly = b 


# ##### ii) Write the pseudocode for the backward substitution corresponding to an upper
# ##### triangular banded matrix of band 2, i.e., for solving U x= b where b is an n × 1 vector
# ##### and U is an n × n upper triangular matrix such that
# ##### U (j, k) = 0, ∀ 1 ≤ j, k ≤ n with k− j > 2.
# ##### The input for the pseudocode are the matrix U and the vector b; the output is the
# ##### vector y.
# ##### What is the operation count for this?

# In[ ]:


#BACKWARD SUBSTITUTION - BAND 2
def backward_solver(U,b):
    n = U.shape[0]
    y = np.zeros(n)
    y[n-1] = b[n-1]/U[n-1,n-1] #Last element of the index 
    if n>=2:
        y[n-2] = (b[n-2] -(U[n-2,n-1] * y[n-1]))/ U[n-2,n-2]
    for i in range(n-3,-1,-1):
        y[i] = (b[i] - ( U[i,i+1] * y[i+1] + U[i,i+2] * y[i+2]))/U[i,i]
    return y


# In[ ]:


#TEST
U = np.array([[1,1,4,0],[0,3,2,1],[0,0,3,3],[0,0,0,2]])
b2 = np.array([1,3,3,3])
print(U)


# In[ ]:


#TEST 2
y = backward_solver(U,b2)
print(y)
print(U@y,b2)  #Uy = b2


# In[ ]:


#WHAT IS THE OPERATION COUNT?
def opcount_forward_band2(n: int):
    if n <= 0:
        return {"mult": 0, "sub": 0, "div": 0, "total": 0}

    mult = 0
    sub = 0
    div = 0

    # j=1
    div += 1

    if n >= 2:
        # j=2
        mult += 1
        sub += 1
        div += 1

    if n >= 3:
        # j=3..n
        mult += 2 * (n - 2)
        sub  += 2 * (n - 2)
        div  += 1 * (n - 2)

    total = mult + sub + div
    return {"mult": mult, "sub": sub, "div": div, "total": total}
    


# In[ ]:


print(opcount_forward_band2(5))
# -> {'mult': 7, 'sub': 7, 'div': 5, 'total': 19}


# ### Operation count (forward substitution, band-2)
# 
# $$
# y_1=\frac{b_1}{L_{11}},\qquad
# y_2=\frac{b_2-L_{21}y_1}{L_{22}},\qquad
# y_j=\frac{b_j-L_{j,j-1}y_{j-1}-L_{j,j-2}y_{j-2}}{L_{jj}},\ \ j=3,\dots,n.
# $$
# 
# $$
# \begin{aligned}
# j=1 &: \ 1\ \text{div},\\
# j=2 &: \ 1\ \text{mult} + 1\ \text{sub} + 1\ \text{div},\\
# j=3,\dots,n &: \ 2\ \text{mult} + 2\ \text{sub} + 1\ \text{div}\ \text{(per row)}.
# \end{aligned}
# $$
# 
# $$
# \#\text{mult}=1+2(n-2)=2n-3,\qquad
# \#\text{sub}=1+2(n-2)=2n-3,\qquad
# \#\text{div}=1+1+(n-2)=n.
# $$
# 
# $$
# \#\text{ops}=(2n-3)+(2n-3)+n=5n-6=O(n).
# $$
# 
# ### Same count for backward substitution (band-2)
# 
# $$
# x_n=\frac{b_n}{U_{nn}},\qquad
# x_{n-1}=\frac{b_{n-1}-U_{n-1,n}x_n}{U_{n-1,n-1}},\qquad
# x_j=\frac{b_j-U_{j,j+1}x_{j+1}-U_{j,j+2}x_{j+2}}{U_{jj}},\ \ j=n-2,\dots,1.
# $$
# 
# $$
# \#\text{ops}=5n-6=O(n).
# $$
