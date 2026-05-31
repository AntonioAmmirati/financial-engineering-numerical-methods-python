#!/usr/bin/env python
# coding: utf-8

# ### HW4 - (2)
# This problem refers to the data from Table 11-1 on Page 322 of Salih Neftci’s Financial
# Engineering book.
# arbitrage–free.
# Consider the market and states from that example. We know this market is complete and
# (i) For each of the other securities in the table, find and report Vmarket, the unique market
# value of the security in this complete market.

# In[ ]:


import numpy as np


# We calibrate the state-price vector $Q$ using the six liquid benchmark options from Neftci’s example ($1200C$, $1275C$, $1350C$, $1200P$, $1050P$, $900P$), and then price all remaining securities in Table 11-1 as $V_{\text{market}} = z^\top Q$.

# We choose the representative state values for the liquid-strike intervals as
# $w_2,w_3,w_4,w_5=(975,\;1125,\;1237.5,\;1312.5)$,
# corresponding to the midpoints of the intervals induced by the most liquid option strikes.
# 
# For the boundary states, we follow Neftci approach, we set
# $w_1=750$ and $w_6=1400$.
# 
# Hence, the state representatives are
# $$(w_1,w_2,w_3,w_4,w_5,w_6)=(750,\;975,\;1125,\;1237.5,\;1312.5,\;1400).$$

# In[ ]:


w = [750,975,1125,1237.5,1312.5,1400]


# In[ ]:


#Payoff Matrix(Mt)
M = np.array([[0,0,0,w[3] - 1200 , w[4] - 1200 , w[5] -1200],#Call K = 1200
     [0,0,0,0,w[4] - 1275 , w[5] - 1275],#Call K = 1275
     [0,0,0,0,0,w[5]-1350], #Call K = 1350
     [900-w[0],0,0,0,0,0], #Put K = 900
     [1050 - w[0],1050 - w[1] , 0 ,0,0,0], # Put K = 1050
     [1200 - w[0],1200-w[1] , 1200 -w[2],0,0,0]]) # Put K = 1200


# In[ ]:


print(M)


# In[ ]:


# St = MtQ
#so Q = forward + backward solver for M(6,6) since M(3,3) is upper triangular and the successive M(3,3) is lower triangular.


# In[ ]:


M_upper = np.array([[w[3] - 1200 , w[4] - 1200 , w[5] -1200],#Call K = 1200
     [0,w[4] - 1275 , w[5] - 1275],#Call K = 1275
     [0,0,w[5]-1350]]) #Call K = 1350
M_lower = np.array([ [900-w[0],0,0], #Put K = 900
     [1050 - w[0],1050 - w[1] , 0 ], # Put K = 1050
     [1200 - w[0],1200-w[1] , 1200 -w[2]]]) # Put K = 1200
print(M_upper)
print(M_lower)


# To reconstruct $V_{\text{market}}$ for each option, we first calibrate the state-price vector $Q$ using the six liquid benchmark options from Neftci’s example:
# $1200C, 1275C, 1350C, 900P, 1050P, 1200P$.
# 
# Let $M_{\text{base}}$ be the $6\times 6$ payoff matrix of these benchmark securities across the six chosen states, and let $St_{\text{mid,base}}$ be their midpoint price vector, where each midpoint is computed as
# $V_{\text{mid}} = (\text{Bid}+\text{Ask})/2$.
# 
# We then solve the linear system
# $M_{\text{base}} Q = St_{\text{mid,base}}$
# to obtain the unique state-price vector $Q$ (under completeness).
# 
# For any other security with payoff vector $z$, its unique market value in this complete market is reconstructed as
# $V_{\text{market}} = z^\top Q$.
# 
# Equivalently, if $M_{\text{other}}$ collects the payoff vectors of all remaining securities, then their reconstructed market values are
# $V_{\text{market}} = M_{\text{other}} Q$.

# Using the six benchmark securities in the same order as the payoff matrix rows
# $(1200C,\;1275C,\;1350C,\;900P,\;1050P,\;1200P)$,
# the midpoint-price vector is
# 
# $$
# p_{\text{mid}}=
# \begin{bmatrix}
# 53.8\\
# 22.3\\
# 6.8\\
# 3.75\\
# 14.75\\
# 54.9
# \end{bmatrix}.
# $$
# 
# We then recover the state-price vector $Q$ by solving
# $$
# M Q = St_{\text{mid}}.
# $$

# In[ ]:


#Now we decompose the vector p in 2 parts , one for solving the upper triangular matrix and another to solve the lower triangular matrix
St_upper = np.array([53.8,22.3,6.8])
St_lower = np.array([3.75,14.75,54.9])
St_ = np.array([53.8,22.3,6.8,3.75,14.75,54.9])


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


Q_up = backward_solver(M_upper , St_upper)
Q_low = forward_solver(M_lower,St_lower)


# In[ ]:


Q_ = np.concatenate((Q_low,Q_up))
print(Q_)


# In[ ]:


np.allclose(M @ Q_, St_)


# `np.allclose(M @ Q_, St_)` checks whether the model-implied prices $M Q_$ match the observed price vector $S_t$ (up to numerical tolerance); returning `True` confirms that the computed state-price vector $Q_$ correctly reproduces the benchmark option prices (i.e., the linear system $M Q = S_t$ is satisfied).

# In[ ]:


#Now we find the other instruments payoff matrix.
#w = [750,975,1125,1237.5,1312.5,1400]
M_other = np.array([[0,0,0,w[3] -1175,w[4] - 1175 ,w[5] - 1175],#Call K=1175
                    [0,0,0,w[3]-1225,w[4]-1225,w[5]-1225],#Call K = 1225
                    [0,0,0,0,w[4]-1250,w[5]-1250], #Call K=1250
                    [0,0,0,0,w[4]-1300,w[5]-1300], #Call K = 1300
                    [0,0,0,0,0,w[5]-1325], #Call K = 1325
                    [0,0,0,0,0,w[5]-1375], # Call K = 1375
                    [0,0,0,0,0,0],#Call K = 1400
                    [0,0,0,0,0,0], #Call K =1425
                    [0,0,0,0,0,0], #Call K = 1450
                    [0,0,0,0,0,0],#Call K = 1475
                    [800-w[0],0,0,0,0,0], #Put K = 800
                    [950-w[0],0,0,0,0,0], #Put K = 950
                    [995-w[0],995-w[1],0,0,0,0],#Put K = 995
                    [1025-w[0],1025-w[1],0,0,0,0], #Put K = 1025
                    [1060-w[0],1060-w[1],0,0,0,0], #Put K = 1060
                    [1075-w[0],1075-w[1],0,0,0,0],#Put K = 1075
                    [1100-w[0],1100-w[1],0,0,0,0],#Put K = 1100
                    [1150-w[0],1150-w[1],1150 - w[2],0,0,0], #Put K = 1150
                    [1175-w[0],1175-w[1],1175 -w[2],0,0,0]]) #Put k = 1175


# In[ ]:


print(M_other)


# In[ ]:


V_market_other = M_other @ Q_


# In[ ]:


print(V_market_other)


# ### Tail strikes outside the modeled state grid (6-state market)
# 
# In the 6-state discretization we use the representative state values
# $(w_1,\dots,w_6)=(750,975,1125,1237.5,1312.5,1400)$.
# For a call option with strike $K>w_6$ (or $K\ge w_6$ with $(w_6-K)^+=0$), the payoff is zero in every modeled state:
# $$
# z^{(call)}(K) = \big((w_1-K)^+,\dots,(w_6-K)^+\big) = (0,0,0,0,0,0).
# $$
# Therefore, the model-implied price in this discretized market is
# $$
# V_{\text{market}}(K)=z^{(call)}(K)^\top Q = 0.
# $$
# This highlights a limitation of the 6-state grid: the right tail beyond $w_6$ is not represented, so high-strike calls are priced at zero by construction.

# In[ ]:


#relative error
V_mid_other = np.array([
    69.0, 41.3, 30.6, 15.6, 10.5, 4.35,2.85,1.625,1.025,0.575,
    1.425, 5.8, 9.0, 11.85, 16.45, 18.75, 23.7, 36.3, 45.1
]) #Midpoint bid-ask((bid+ask)/2) of the contracts. 

rel_err = np.abs(V_market_other - V_mid_other) / V_mid_other
avg_abs_err = rel_err.mean()

print(rel_err)
print(avg_abs_err)


# ### Discussion (6-state market)
# 
# The overall average absolute relative error is approximately **26%**, which is mainly driven by the coarse **6-state discretization**. With only a small number of representative states, the model provides a rough approximation of the risk-neutral distribution of $S(T)$, especially in the tails.
# 
# In particular, the right tail is **truncated** at the largest representative state $w_6=1400$. As a consequence, high-strike calls with $K \ge w_6$ have zero payoff in all modeled states, implying
# $$
# V_{\text{market}} = z^\top Q = 0,
# $$
# even though their observed midpoint prices are strictly positive. This produces relative errors close to 100% for those contracts and significantly inflates the average error.
# 
# More generally, payoffs for far out-of-the-money options are concentrated on a very small subset of extreme states, so their prices (and the implied state prices) are more sensitive to discretization error and bid–ask noise. Increasing the number of states will reduces tail truncation and improves the overall fit.

# ##### (iii)  Create new a market with the following instruments:
# {1200−Call, 1275−Call, 1350−Call, 1425−Call, 1200−P ut, 1050−P ut, 950−P ut, 800−P ut}.
# There will be 8 states of the world as follows:
# ω1 S(T ) = 650
# ω2 S(T ) = 875
# ω3 S(T ) = 1000
# ω4 S(T ) = 1125
# ω5 S(T ) = 1237.50
# ω6 S(T ) = 1312.50
# ω7 S(T ) = 1387.50
# ω8 S(T ) = 1500

# In[ ]:


w_ = [650,875,1000,1125,1237.50,1312.50,1387.50,1500]


# #### (1) What is the payoﬀ matrix?

# In[ ]:


M_ = np.array([[0,0,0,0,w_[4] - 1200 , w_[5] - 1200 , w_[6] - 1200,w_[7]-1200],#Call K = 1200
               [0,0,0,0,0,w_[5] - 1275 , w_[6] - 1275,w_[7]-1275],#Call K = 1275
               [0,0,0,0,0,0,w_[6] - 1350,w_[7]-1350],#Call K = 1350
               [0,0,0,0,0,0,0,w_[7]-1425], # Call K = 1425
               [800 - w_[0],0,0,0,0,0,0,0],#Put K= 800
               [950 - w_[0] , 950 - w_[1],0,0,0,0,0,0], #Put K = 950
               [1050 - w_[0],1050-w_[1],1050-w_[2],0,0,0,0,0],#Put K = 1050
               [1200 - w_[0],1200 -w_[1],1200-w_[2],1200-w_[3],0,0,0,0]])#Put K = 1200


# ### For convenience, we reorder the put securities from $(1200P,1050P,950P,800P)$ to $(800P,950P,1050P,1200P)$ in order to make the put-payoff block explicitly lower triangular. This is only a permutation of rows (i.e., a reordering of securities), so it does not change the linear-algebraic or economic conclusions of the analysis. In particular, it does **not** affect:
# - the rank of the payoff matrix,
# - invertibility (up to row permutations),
# - non-redundancy of the securities,
# - market completeness.
# - the sign pattern of the state-price vector $Q$
# 
# ### The reordering is used purely to make the block structure of the payoff matrix easier to read and to simplify the determinant/rank argument.

# In[ ]:


print(M_)


# #### (2) Are these securities non–redundant?

# #### Yes. These securities are non-redundant because their payoff vectors are linearly independent. Equivalently, the payoff matrix $M$ has full rank:
# #### $$\operatorname{rank}(M)=8.$$ Hence, no security in the set can be replicated exactly as a linear combination of the others.

# ##### (3) Is the market complete?

# To verify non-redundancy (and completeness), we decompose the payoff matrix into the call block and the put block. In our ordering, the call block is upper triangular and the put block is lower triangular.
# 
# Hence, the determinant of each block is the product of its diagonal entries, and the determinant of the full payoff matrix (up to the block-ordering sign, which is positive here since the blocks are $4\times4$) is
# $$
# \det(M)=\det(U_{\text{calls}})\det(L_{\text{puts}}).
# $$
# 
# Since both blocks have nonzero diagonal entries, we have
# $$
# \det(U_{\text{calls}})\neq 0,\qquad \det(L_{\text{puts}})\neq 0,
# $$
# and therefore
# $$
# \det(M)\neq 0.
# $$
# 
# Thus, $M$ is invertible, $\operatorname{rank}(M)=8$, the securities are non-redundant, and the market is complete.

# In[ ]:


#Now we compute the determinants of both matrices.
#First we decompose the matrix M_
M_upper_calls = np.array([[w_[4] - 1200 , w_[5] - 1200 , w_[6] - 1200,w_[7]-1200],
               [0,w_[5] - 1275 , w_[6] - 1275,w_[7]-1275],
               [0,0,w_[6] - 1350,w_[7]-1350],
               [0,0,0,w_[7]-1425]]) #Upper triangular Matrix (Calls)
M_lower_puts = np.array([  [800 - w_[0],0,0,0],
               [950 - w_[0] , 950 - w_[1],0,0], 
               [1050 - w_[0],1050-w_[1],1050-w_[2],0],
               [1200 - w_[0],1200 -w_[1],1200-w_[2],1200-w_[3]]]) #Lower triangular (Puts)


# In[ ]:


print(M_upper_calls)


# In[ ]:


print(M_lower_puts)


# In[ ]:


#determinants
det_calls = np.linalg.det(M_upper_calls) #equivalently diag(i)..*diag(n)
det_puts = np.linalg.det(M_lower_puts)
det_M = det_calls * det_puts
print(det_M)


# ##### Since $\det(M)\neq 0$, , hence the matrix is non-singular and the market is complete

# ##### (4) Is the market arbitrage–free?

# To determine whether the market is arbitrage-free, we compute the state-price vector $Q$ by solving
# $$
# M Q = St,
# $$
# where $St$ is the vector of observed market prices (here, midpoint bid-ask prices) of the eight selected securities.
# 
# Since we have already shown that the market is complete (the payoff matrix $M$ has full rank), the solution $Q$ is unique. In this finite-state complete-market framework, the market is arbitrage-free if and only if all state prices are strictly positive:
# $$
# Q_i>0 \quad \text{for all } i=1,\dots,8.
# $$
# 
# Therefore, after computing $Q$, we check the sign of each component. If all entries are positive, the market is arbitrage-free; otherwise, it is not.

# In[ ]:


#Following Table 11-1 of Neftci book, we compute the mid-prices(bid+ask/2) of the 8 contracts:
St = np.array([53.8, 22.3, 6.8, 1.625, 1.425, 5.8, 14.75, 54.9])


# #### Now we find Q , using backward and forward substitution on the two matrices(calls payoff(upper triangular), and puts payoff(lower triangular)) 

# In[ ]:


#let's decompose St in:
St_up = np.array([53.8, 22.3, 6.8, 1.625])
St_low = np.array([1.425, 5.8, 14.75, 54.9])


# In[ ]:


#forward and backward substitution:
Q_up = backward_solver(M_upper_calls,St_up)
Q_low = forward_solver(M_lower_puts,St_low)
Q = np.concatenate((Q_low,Q_up))


# In[ ]:


print(Q)


# Solving the pricing system $M Q = S_t$ yields the state-price vector $Q=(Q_1,\dots,Q_8)^\top$.  
# In this finite-state complete-market framework, the market is arbitrage-free if and only if all state prices are strictly positive. Since
# $$
# Q_i>0 \quad \text{for all } i=1,\dots,8,
# $$
# the market is arbitrage-free.

# In[ ]:


np.allclose(M_ @ Q, St)


# ##### (5) Price all the other instruments in the market using the values of the eight securities above. Compute for each security the relative approximation error as above, as well as the overall average approximation error. Comment on the results.

# In[ ]:


w_ = [650,875,1000,1125,1237.50,1312.50,1387.50,1500]
M_others_ = np.array([[0,0,0,0,w_[4] - 1175,w_[5] - 1175,w_[6] - 1175 , w_[7] - 1175],#Call K = 1175
                      [0,0,0,0,w_[4] - 1225,w_[5] - 1225,w_[6] - 1225 , w_[7] - 1225],#Call K = 1225
                      [0,0,0,0,0,w_[5] - 1250,w_[6] - 1250 , w_[7] - 1250],#Call K = 1250
                      [0,0,0,0,0,w_[5] - 1300,w_[6] - 1300 , w_[7] - 1300],#Call K = 1300
                      [0,0,0,0,0,0,w_[6] - 1325 , w_[7] - 1325],#Call K = 1325
                      [0,0,0,0,0,0,w_[6] - 1375 , w_[7] - 1375],#Call K = 1375
                      [0,0,0,0,0,0,0 , w_[7] - 1400],#Call K = 1400
                      [0,0,0,0,0,0,0 , w_[7] - 1450],#Call K = 1450
                      [0,0,0,0,0,0,0 , w_[7] - 1475],#Call K = 1475
                      [900 - w_[0] ,900 - w_[1],0,0,0,0,0,0],#Put K =900
                      [995-w_[0],995-w_[1],0,0,0,0,0,0], #Put K = 995
                      [1025 - w_[0],1025-w_[1],1025-w_[2],0,0,0,0,0], #Put K = 1025
                      [1060 - w_[0],1060 -w_[1],1060-w_[2],0,0,0,0,0], # Put K = 1060
                      [1075 - w_[0],1075 - w_[1],1075 -w_[2],0,0,0,0,0], # Put K = 1075
                      [1100 - w_[0],1100 - w_[1],1100 -w_[2],0,0,0,0,0], # Put K = 1100

                    
                      [1150 - w_[0],1150 - w_[1],1150 -w_[2],1150 -w_[3],0,0,0,0], # Put K = 1150
                      [1175 - w_[0],1175 - w_[1],1175 -w_[2],1175 -w_[3],0,0,0,0]]) # Put K = 1175
                      
                      


# In[ ]:


print(M_others_)


# In[ ]:


M_others_.shape[0]


# In[ ]:


# Now we find St(contract price vector at time t) using Q


# In[ ]:


St_os = M_others_ @ Q


# In[ ]:


St_os


# In[ ]:


Vmid_other = np.array([
    69.0, 41.3, 30.6, 15.6, 10.5, 4.35, 2.85, 1.025, 0.575,
    3.75, 9.0, 11.85, 16.45, 18.75, 23.7, 36.3, 45.1
])


# In[ ]:


#relative error (2)
rel_err_ = np.abs(St_os - Vmid_other) / Vmid_other
avg_abs_err_ = rel_err_.mean()

print(rel_err_)
print(avg_abs_err_)


# ### (5) Pricing of the remaining instruments and approximation errors
# 
# Using the state-price vector $Q$ implied by the 8-security complete market, we price each remaining instrument as
# $$
# V_{\text{market}} = z^\top Q,
# $$
# and compare it to the observed midpoint price
# $$
# V_{\text{mid}}=\frac{\text{Bid}+\text{Ask}}{2}.
# $$
# The relative approximation error is
# $$
# \text{RelErr}=\frac{|V_{\text{market}}-V_{\text{mid}}|}{V_{\text{mid}}}.
# $$
# 
# #### Results table (other instruments)
# | Instrument | $V_{\text{market}}$ | $V_{\text{mid}}$ | RelErr |
# |---|---:|---:|---:|
# | 1175C | 67.3750 | 69.0000 | 0.0236 |
# | 1225C | 40.2250 | 41.3000 | 0.0260 |
# | 1250C | 29.7250 | 30.6000 | 0.0286 |
# | 1300C | 14.8750 | 15.6000 | 0.0465 |
# | 1325C | 9.7083 | 10.5000 | 0.0754 |
# | 1375C | 3.8917 | 4.3500 | 0.1054 |
# | 1400C | 2.1667 | 2.8500 | 0.2398 |
# | 1450C | 1.0833 | 1.0250 | 0.0569 |
# | 1475C | 0.5417 | 0.5750 | 0.0580 |
# | 900P  | 3.3583 | 3.7500 | 0.1044 |
# | 995P  | 7.9975 | 9.0000 | 0.1114 |
# | 1025P | 11.4958 | 11.8500 | 0.0299 |
# | 1060P | 16.0517 | 16.4500 | 0.0242 |
# | 1075P | 18.0042 | 18.7500 | 0.0398 |
# | 1100P | 21.2583 | 23.7000 | 0.1030 |
# | 1150P | 34.6417 | 36.3000 | 0.0457 |
# | 1175P | 44.7708 | 45.1000 | 0.0073 |
# 
# **Average absolute relative error (across all 17 instruments):**
# $$
# \overline{\text{RelErr}} = 0.0662 \quad (\text{about } 6.62\%).
# $$
# 
# #### Qualitative discussion
# Overall, the approximation quality is good: the average relative error is about **6.6%**, and many strikes are priced within a few percent of the midpoint quotes (especially around the more central strikes).
# 
# The largest discrepancy occurs for the **1400-call** (about **24%** relative error). This is consistent with the discretized 8-state representation: the far-right tail is represented by only a small number of high states, so deep out-of-the-money calls depend heavily on a coarse tail approximation and can show larger relative errors.
# 
# For puts, errors are generally moderate (often in the 2–11% range). The fit tends to be better for strikes closer to the central region covered by multiple states, and weaker for contracts whose payoff is concentrated on a small subset of extreme states.

# Finally, compared to the 6-state discretization, the 8-state market reduces right-tail truncation (by extending the highest representative state to $S(T)=1500$) and therefore provides a more realistic pricing approximation for higher-strike call options.
