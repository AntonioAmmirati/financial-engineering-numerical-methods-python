#!/usr/bin/env python
# coding: utf-8

# ##### Exercize 2 : The file S&P500 ETF Option 0917.xlsx contains the S&P 500 option prices with 9/29/2017 maturity as of March 16, 2017. The spot price of the index corresponding to these option prices was 2, 381.

# In[ ]:


import pandas as pd
import numpy as np


# ### Initial step: data extraction and construction of market option prices
# 
# We begin by importing the option dataset from the Excel file and cleaning the table so that only valid observations are retained. From the dataset, we extract the strike prices $K$, the bid and ask quotes for call options, and the bid and ask quotes for put options, all for the same maturity.
# 
# For each strike, we do not use the bid or ask quote alone. Instead, we estimate the market option price by taking the midpoint between bid and ask. Thus, the call and put market prices are defined as
# 
# $$
# C_m = \frac{C_{\text{bid}} + C_{\text{ask}}}{2},
# \qquad
# P_m = \frac{P_{\text{bid}} + P_{\text{ask}}}{2}.
# $$
# 
# Using midpoint prices is standard in practice because it provides a more stable proxy for the transaction price than using only the bid or only the ask. In particular, it reduces the effect of the bid–ask spread and gives a representative estimate of the market value of the option.
# 
# Therefore, at this initial stage, we are doing three things:
# 
# 1. importing and cleaning the option data;
# 2. extracting the common strike vector $K$;
# 3. constructing the market call and put prices $C_m$ and $P_m$ from bid–ask midpoints.
# 
# These midpoint prices will then be used throughout the rest of the analysis to estimate $r$ and $q$, compute Black–Scholes implied volatilities, and compare them with the Stefanica–Radoicic approximation.

# In[ ]:


df = pd.read_excel('S_P500_ETF_Option_0917.xlsx')


# In[ ]:


df.columns = ['Calls_K' , 'contract' , 'bid_c' ,'ask_c','volume','Puts_K','contract','bid_p','ask_p','volume_p']
df = df.dropna()
df.head()


# In[ ]:


K = list(df['Calls_K'].dropna())


# In[ ]:


K.pop(0)


# In[ ]:


print(K)
K = np.array(K)


# Let $K$ denote the vector of strike prices.  
# For each strike, the market call and put prices are estimated by the bid–ask midpoints
# 
# $$
# C_m = \frac{C_{\text{bid}} + C_{\text{ask}}}{2},
# \qquad
# P_m = \frac{P_{\text{bid}} + P_{\text{ask}}}{2}.
# $$

# In[ ]:


#Calls midprice
Cm_ask = list(df['ask_c'].dropna())
Cm_ask.pop(0)
Cm_bid = list(df['bid_c'].dropna())
Cm_bid.pop(0)


# In[ ]:


# (ask + bid)/2
Cm_ask = np.array(Cm_ask , dtype = float)
Cm_bid = np.array(Cm_bid , dtype = float)
Cm = (Cm_ask + Cm_bid)/2


# In[ ]:


print(Cm)


# In[ ]:


#Puts midprice
Pm_ask = list(df['ask_p'].dropna())
Pm_ask.pop(0)
Pm_bid = list(df['bid_p'].dropna())
Pm_bid.pop(0)


# In[ ]:


# (ask + bid)/2
Pm_ask = np.array(Pm_ask , dtype = float)
Pm_bid = np.array(Pm_bid , dtype = float)
Pm = (Pm_ask + Pm_bid)/2


# In[ ]:


print(Pm)


# ##### (i) Use a least squares method to compute the annualized continuous dividend yield of the S&P 500 index and for the risk–free rate implied by these prices.

# In[ ]:


b = Cm - Pm
K = K.reshape(-1,1)
ones = np.ones((len(K),1))
A = np.hstack([ones, -K])


# The vector of mid-price differences is
# 
# $$
# b =
# \begin{pmatrix}
# 224.75 \\
# 199.90000534 \\
# 175.14999390 \\
# 150.34999847 \\
# 125.49999428 \\
# 100.64999962 \\
# 75.85000229 \\
# 51.00000000 \\
# 26.25000381 \\
# 1.40000153 \\
# -23.34999847 \\
# -48.20000267 \\
# -73.05000114 \\
# -97.80000687 \\
# -122.65000057 \\
# -171.89999723 \\
# -221.49999714 \\
# -272.09999394 \\
# -321.75000918 \\
# -420.95000610
# \end{pmatrix}.
# $$
# 
# The design matrix is
# 
# $$
# A =
# \begin{pmatrix}
# 1 & -2150 \\
# 1 & -2175 \\
# 1 & -2200 \\
# 1 & -2225 \\
# 1 & -2250 \\
# 1 & -2275 \\
# 1 & -2300 \\
# 1 & -2325 \\
# 1 & -2350 \\
# 1 & -2375 \\
# 1 & -2400 \\
# 1 & -2425 \\
# 1 & -2450 \\
# 1 & -2475 \\
# 1 & -2500 \\
# 1 & -2550 \\
# 1 & -2600 \\
# 1 & -2650 \\
# 1 & -2700 \\
# 1 & -2800
# \end{pmatrix}.
# $$

# ---------
# ### (i) Estimation of $r$ and $q$ by Ordinary Least Squares
# 
# To estimate the risk-free rate $r$ and the dividend yield $q$, we use the put–call parity relation
# 
# $$
# C_m - P_m = S_0 e^{-qT} - K e^{-rT}.
# $$
# 
# Let
# 
# $$
# a = S_0 e^{-qT}, \qquad d = e^{-rT}.
# $$
# 
# Then the equation becomes linear in the unknowns $a$ and $d$:
# 
# $$
# C_m - P_m = a - Kd.
# $$
# 
# For all strikes, this can be written in matrix form as
# 
# $$
# b = Ax,
# $$
# 
# where
# 
# $$
# b =
# \begin{pmatrix}
# C_1 - P_1 \\
# C_2 - P_2 \\
# \vdots \\
# C_n - P_n
# \end{pmatrix},
# \qquad
# A =
# \begin{pmatrix}
# 1 & -K_1 \\
# 1 & -K_2 \\
# \vdots & \vdots \\
# 1 & -K_n
# \end{pmatrix},
# \qquad
# x =
# \begin{pmatrix}
# a \\
# d
# \end{pmatrix}.
# $$
# 
# Since the system is overdetermined, we estimate $x$ by OLS, solving
# 
# $$
# \min_x \|Ax - b\|_2^2.
# $$
# 
# The normal equations are
# 
# $$
# A^\top A x = A^\top b.
# $$
# 
# Thus,
# 
# $$
# \hat{x} =
# \begin{pmatrix}
# \hat{a} \\
# \hat{d}
# \end{pmatrix}
# =
# (A^\top A)^{-1} A^\top b.
# $$
# 
# Finally, we recover $r$ and $q$ from
# 
# $$
# \hat{r} = -\frac{\ln(\hat{d})}{T},
# \qquad
# \hat{q} = -\frac{\ln(\hat{a}/S_0)}{T}.
# $$
# 
# ### Why we use Cholesky decomposition
# 
# To solve the normal equations
# 
# $$
# A^\top A x = A^\top b,
# $$
# 
# we use Cholesky decomposition.
# 
# This is appropriate because the matrix
# 
# $$
# A^\top A
# $$
# 
# is symmetric, and if the columns of $A$ are linearly independent, then it is also positive definite. Therefore, $A^\top A$ is **symmetric positive definite (SPD)**.
# 
# Hence, we can write
# 
# $$
# A^\top A = U^\top U
# $$
# 
# with $U$ upper triangular. Then the linear system is solved in two steps:
# 
# 1. solve
# $$
# U^\top y = A^\top b
# $$
# by forward substitution;
# 
# 2. solve
# $$
# U x = y
# $$
# by backward substitution.
# 
# We use Cholesky because it is efficient, numerically stable for SPD matrices, and exploits the special structure of the matrix $A^\top A$.

# In[ ]:


def cholesky(A):
    n = A.shape[0]
    U = np.zeros((n, n))
    
    for i in range(n):
        
       
        U[i, i] = np.sqrt(A[i, i] - np.sum(U[:i, i]**2))
        
        
        for j in range(i+1, n):
            U[i, j] = (A[i, j] - np.sum(U[:i, i] * U[:i, j])) / U[i, i]
    
    return U


# To solve the normal equations, we use Cholesky decomposition. This is appropriate because the matrix $A^\top A$ is symmetric, and since the columns of $A$ are linearly independent, it is also positive definite. Therefore, $A^\top A$ is symmetric positive definite (SPD), and we can write
# 
# $$
# A^\top A = U^\top U,
# $$
# 
# where $U$ is upper triangular. We then solve
# 
# $$
# U^\top y = A^\top b,
# \qquad
# Ux = y
# $$
# 
# by forward and backward substitution.

# In[ ]:


AtA = A.T @ A
U = cholesky(AtA)
U
b_ = A.T @ b


# In[ ]:


def forward_sub(U,b):
    n = U.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += U[i,j]*y[j]
        y[i] = (b[i] - s)/ U[i,i]
    return y
def backward_sub(U,y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        s = 0.0
        for j in range(i+1,n):
            s+= U[i,j]*x[j]
        x[i] = (y[i]-s)/U[i,i]
    return x   


# In[ ]:


y = forward_sub(U.T,b_)
x = backward_sub(U,y)


# In[ ]:


print(x)


# In[ ]:


#now we extrapolate r and q
T = 197/365
St = 2381
r = - (np.log(x[1]))/ T
q = -(np.log(x[0]/St))/T


# In[ ]:


print(q)


# ##### (ii) Compute the implied volatilities for each option.
# 
# 
# #### For each option price, we compute the implied volatility by solving the nonlinear equation
# 
# $$
# V_{BS}(S_0,K,T,r,q,\sigma)=V_{\text{mkt}},
# $$
# 
# using the Newton–Raphson method, where $V_{BS}$ denotes either the Black–Scholes call or put price, and the derivative with respect to $\sigma$ is the Black–Scholes vega. If the Newton–Raphson method fails to converge, the bisection method may be used as a more robust numerical alternative.
# 
# Starting from an initial guess $\sigma^{(0)}$, the Newton–Raphson iteration is
# 
# $$
# \sigma^{(n+1)} = \sigma^{(n)} - \frac{V_{BS}(S_0,K,T,r,q,\sigma^{(n)}) - V_{\text{mkt}}}{\text{Vega}(S_0,K,T,r,q,\sigma^{(n)})}.
# $$
# 
# The iteration is repeated until convergence, that is, until the pricing error is sufficiently close to zero. If the Newton–Raphson method fails to converge, the bisection method may be used as a more robust numerical alternative.

# In[ ]:


from scipy.stats import norm
def c_bs(sigma,K):
    sigma = float(sigma)
    K = float(K)
    d1 = (np.log(St/K) + (r-q + 0.5*sigma**2)*T)/ (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    c_bs = St*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return c_bs


# In[ ]:


def vega(sigma,K):
     sigma = float(sigma)
     K = float(K)
     d1 = (np.log(St/K) + (r-q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
     vega = St*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)
     return vega


# In[ ]:


initial_guess = 0.1


# In[ ]:


def newton_raphson_c(C_mkt,K,sigma, tol = 1e-6 , max_iter = 50):
    sigma = float(sigma)
    K = float(K)
    sigma = float(sigma)
    for i in range(max_iter):
        f = c_bs(sigma,K) - C_mkt
        sigma_new = sigma - (f /vega(sigma,K))
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new
    return sigma 
        


# In[ ]:


#Implied volatility for each calls with different strike
K = np.array(K).flatten()
Cm = np.array(Cm).flatten()
iv_calls = []
for i in range(len(K)):
    iv_calls.append(newton_raphson_c(Cm[i],K[i],initial_guess))


# In[ ]:


print(iv_calls)


# In[ ]:


def p_bs(sigma,K):
    K = float(K)
    sigma = float(sigma)
    d1 = (np.log(St/K) + (r-q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    p_bs = K*norm.cdf(-d2)*np.exp(-r*T) - St*norm.cdf(-d1)*np.exp(-q*T)
    return p_bs
def newton_raphson_p(K,p_mkt, sigma , tol = 1e-6 , max_iter = 50):
    K = float(K)
    sigma = float(sigma)
    p_mkt = float(p_mkt)
    for i in range(max_iter):
        f = p_bs(sigma,K) - p_mkt 
        sigma_new = sigma - (f/vega(sigma,K))
        if np.abs(sigma_new-sigma) < tol:
            return sigma_new
        sigma = sigma_new
    return sigma 


# In[ ]:


#Implied volatility for puts
iv_puts=[]
for i in range(len(K)):
    iv_puts.append(newton_raphson_p(K[i],Pm[i],initial_guess))


# In[ ]:


print(iv_puts)


# In[ ]:


iv_calls = np.array(iv_calls)
iv_puts = np.array(iv_puts)

abs_diff = np.abs(iv_calls - iv_puts)
print(abs_diff)
print("max diff =", abs_diff.max())
print("mean diff =", abs_diff.mean())


# ### Point 2: Black–Scholes implied volatilities from call and put prices
# 
# In this part, we compute the implied volatility for each strike by inverting the Black–Scholes formula separately for call prices and put prices. More precisely, for each market price we solve
# 
# $$
# C_{BS}(S_0,K,T,r,q,\sigma)=C_{\text{mkt}}
# $$
# 
# for calls, and
# 
# $$
# P_{BS}(S_0,K,T,r,q,\sigma)=P_{\text{mkt}}
# $$
# 
# for puts, using the Newton–Raphson method.
# 
# After computing the two implied volatility vectors, we compare them strike by strike through the absolute difference
# 
# $$
# \lvert \sigma_{\text{call}} - \sigma_{\text{put}} \rvert.
# $$
# 
# This comparison is useful because, when the inputs are consistent with put–call parity, the implied volatilities extracted from calls and puts at the same strike and maturity should be very close.
# 
# The numerical results confirm this. The absolute differences are very small across all strikes, with
# 
# $$
# \max \lvert \sigma_{\text{call}} - \sigma_{\text{put}} \rvert \approx 0.00427,
# $$
# 
# and
# 
# $$
# \text{mean } \lvert \sigma_{\text{call}} - \sigma_{\text{put}} \rvert \approx 0.00068.
# $$
# 
# This indicates that the implied volatilities extracted from calls and puts are highly consistent overall.
# 
# A compact summary is given below:
# 
# | Quantity | Value |
# |---|---:|
# | Maximum absolute difference | $0.00427$ |
# | Mean absolute difference | $0.00068$ |
# 
# Therefore, the call-based and put-based implied volatilities are almost identical for most strikes, which suggests that the option quotes are internally consistent and that the numerical inversion procedure is working properly.

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(K, iv_calls * 100, color='blue', label='Calls')
plt.plot(K, iv_puts * 100, color='red', label='Puts', linestyle='--')
plt.xlabel('Strike')
plt.ylabel('Implied Vol %')
plt.title('Volatility Skew SPX - Sep 2017')
plt.legend()
plt.grid(True)
plt.show()


# ### (iii) Comparison between call and put implied volatilities
# 
# To compare the implied volatilities of calls and puts with the same strike, we compute
# 
# $$
# \left| \sigma_{\text{call}} - \sigma_{\text{put}} \right|.
# $$
# 
# The differences are very small across all strikes. In particular,
# 
# $$
# \max \left| \sigma_{\text{call}} - \sigma_{\text{put}} \right| \approx 0.00427,
# $$
# 
# and
# 
# $$
# \text{mean} \left| \sigma_{\text{call}} - \sigma_{\text{put}} \right| \approx 0.00068.
# $$
# 
# Therefore, the implied volatilities extracted from calls and puts are highly consistent overall.

# -------
# 
# #### (iv) Use the explicit implied volatility formulas from the paper “An Explicit Implied Volatility Formula” from https://papers.ssrn.com/sol3/papers.cfm?abstract id=2908494 (see Tables 1 and 2 for the pseudocodes) to compute approximate values for the implied volatilites of all these options. Report these approximate values and the relative errors with respect to the corresponding Black–Scholes implied volatilities computed in part (ii).
# 
# ---- 

# ### Step 1: Define the auxiliary function $A(x)$
# 
# We first define the auxiliary function $A(x)$ used in the Stefanica–Radoicic approximation formula.

# In[ ]:


def A_func(x):
    if x >= 0:
        return 0.5 + 0.5 * np.sqrt(1 - np.exp(-2 * x**2 / np.pi))
    else:
        return 0.5 - 0.5 * np.sqrt(1 - np.exp(-2 * x**2 / np.pi))


# ### Step 2: Compute the forward price $F$ and log-moneyness $y$
# 
# For each strike $K_i$, we compute
# 
# $$
# F = S_t e^{(r-q)T},
# \qquad
# y = \ln\left(\frac{F}{K_i}\right).
# $$
# 
# These quantities are needed in the explicit approximation formula.

# In[ ]:


F = St * np.exp((r - q) * T)
print("F =", F)


# ### Step 3: Compute the approximation for implied volatility from call prices
# 
# For each strike, we compute
# 
# $$
# \alpha_C = \frac{C_m}{K e^{-rT}},
# \qquad
# R = 2\alpha_C - e^y + 1,
# $$
# 
# then the coefficients $A$, $B$, $C$, followed by
# 
# $$
# \beta = \frac{2C}{B + \sqrt{B^2 + 4AC}},
# \qquad
# \gamma = -\frac{\pi}{2}\ln(\beta).
# $$
# 
# Finally, we choose the correct branch of the formula depending on the sign of $y$ and the comparison between $C_m$ and $C_0$.

# In[ ]:


iv_calls_approx = []

for i in range(len(K)):
    y = np.log(F / K[i])
    
    if y >= 0:
        # K <= F: call ITM-forward → Table 1 applicata alla call
        alpha_c = Cm[i] / (K[i] * np.exp(-r * T))
        R = 2 * alpha_c - np.exp(y) + 1
        
        A_coef = (np.exp((1 - 2/np.pi)*y) - np.exp(-(1 - 2/np.pi)*y))**2
        B_coef = 4*(np.exp((2/np.pi)*y) + np.exp(-(2/np.pi)*y)) \
               - 2*np.exp(-y)*(np.exp((1-2/np.pi)*y) + np.exp(-(1-2/np.pi)*y)) \
               * (np.exp(2*y) + 1 - R**2)
        C_coef = np.exp(-2*y)*(R**2 - (np.exp(y)-1)**2)*((np.exp(y)+1)**2 - R**2)
        
        beta = 2*C_coef / (B_coef + np.sqrt(B_coef**2 + 4*A_coef*C_coef))
        gamma = -(np.pi/2)*np.log(beta)
        
        # Table 1, caso y >= 0
        C0 = K[i]*np.exp(-r*T)*(np.exp(y)*A_func(np.sqrt(2*y)) - 0.5)
        if Cm[i] <= C0:
            sigma = (np.sqrt(gamma+y) - np.sqrt(gamma-y)) / np.sqrt(T)
        else:
            sigma = (np.sqrt(gamma+y) + np.sqrt(gamma-y)) / np.sqrt(T)
    
    else:
        # K > F: call OTM-forward → per Lemma 4 e p.17
        # IV call OTM = IV put ITM → applica Table 2 alla put di mercato
        alpha_p = Pm[i] / (K[i] * np.exp(-r * T))
        R = 2*alpha_p + np.exp(y) - 1
        
        A_coef = (np.exp((1 - 2/np.pi)*y) - np.exp(-(1 - 2/np.pi)*y))**2
        B_coef = 4*(np.exp((2/np.pi)*y) + np.exp(-(2/np.pi)*y)) \
               - 2*np.exp(-y)*(np.exp((1-2/np.pi)*y) + np.exp(-(1-2/np.pi)*y)) \
               * (np.exp(2*y) + 1 - R**2)
        C_coef = np.exp(-2*y)*(R**2 - (np.exp(y)-1)**2)*((np.exp(y)+1)**2 - R**2)
        
        beta = 2*C_coef / (B_coef + np.sqrt(B_coef**2 + 4*A_coef*C_coef))
        gamma = -(np.pi/2)*np.log(beta)
        
        # Table 2, caso y < 0
        P0 = K[i]*np.exp(-r*T)*(A_func(np.sqrt(-2*y)) - np.exp(y)/2)
        if Pm[i] <= P0:
            sigma = (-np.sqrt(gamma+y) + np.sqrt(gamma-y)) / np.sqrt(T)
        else:
            sigma = (np.sqrt(gamma+y) + np.sqrt(gamma-y)) / np.sqrt(T)
    
    iv_calls_approx.append(sigma)

iv_calls_approx = np.array(iv_calls_approx)

rel_err_calls = np.abs(iv_calls_approx - iv_calls) / iv_calls
print("max rel error calls:", np.max(rel_err_calls))
print("mean rel error calls:", np.mean(rel_err_calls))


# ### Step 4: Compute the approximation for implied volatility from put prices
# 
# We repeat the same procedure for puts. In this case,
# 
# $$
# \alpha_P = \frac{P_m}{K e^{-rT}},
# \qquad
# R = 2\alpha_P + e^y - 1.
# $$
# 
# Then we compute the same coefficients and apply the appropriate branch depending on the sign of $y$ and the comparison between $P_m$ and $P_0$.

# In[ ]:


iv_puts_approx = []

for i in range(len(K)):
    
    y = np.log(F / K[i])
    alpha_p = Pm[i] / (K[i] * np.exp(-r * T))
    R = 2 * alpha_p + np.exp(y) - 1
    
    A_coef = (np.exp((1 - 2/np.pi) * y) - np.exp(-(1 - 2/np.pi) * y))**2
    
    B_coef = 4 * (np.exp((2/np.pi) * y) + np.exp(-(2/np.pi) * y)) \
             - 2 * np.exp(-y) * (np.exp((1 - 2/np.pi) * y) + np.exp(-(1 - 2/np.pi) * y)) * (np.exp(2*y) + 1 - R**2)
    
    C_coef = np.exp(-2*y) * (R**2 - (np.exp(y) - 1)**2) * ((np.exp(y) + 1)**2 - R**2)
    
    beta = 2 * C_coef / (B_coef + np.sqrt(B_coef**2 + 4 * A_coef * C_coef))
    gamma = -(np.pi / 2) * np.log(beta)
    
    if y >= 0:
        P0 = K[i] * np.exp(-r * T) * (0.5 - np.exp(y) * A_func(-np.sqrt(2*y)))
        
        if Pm[i] <= P0:
            sigma = (np.sqrt(gamma + y) - np.sqrt(gamma - y)) / np.sqrt(T)
        else:
            sigma = (np.sqrt(gamma + y) + np.sqrt(gamma - y)) / np.sqrt(T)
    
    else:
        P0 = K[i] * np.exp(-r * T) * (A_func(np.sqrt(-2*y)) - np.exp(y)/2)
        
        if Pm[i] <= P0:
            sigma = (-np.sqrt(gamma + y) + np.sqrt(gamma - y)) / np.sqrt(T)
        else:
            sigma = (np.sqrt(gamma + y) + np.sqrt(gamma - y)) / np.sqrt(T)
    
    iv_puts_approx.append(sigma)

iv_puts_approx = np.array(iv_puts_approx)
print(iv_puts_approx)


# ### Step 5: Compare the explicit approximation with the Black–Scholes implied volatilities
# 
# We now compare the approximate implied volatilities with the Black–Scholes implied volatilities computed previously via Newton–Raphson.
# 
# We use the relative error
# 
# $$
# \text{relative error} = \frac{|\sigma_{\text{approx}} - \sigma_{\text{BS}}|}{\sigma_{\text{BS}}}.
# $$

# In[ ]:


rel_err_calls = np.abs(iv_calls_approx - iv_calls) / iv_calls
rel_err_puts = np.abs(iv_puts_approx - iv_puts) / iv_puts

print("max relative error calls =", np.max(rel_err_calls))
print("mean relative error calls =", np.mean(rel_err_calls))

print("max relative error puts =", np.max(rel_err_puts))
print("mean relative error puts =", np.mean(rel_err_puts))


# ### Step 6: Display the final comparison table
# 
# Finally, we collect the Black–Scholes implied volatilities, the Stefanica–Radoicic approximations, and the relative errors in a single table.

# In[ ]:


results = pd.DataFrame({
    "K": K,
    "iv_call_BS": iv_calls,
    "iv_call_approx": iv_calls_approx,
    "rel_err_call": rel_err_calls,
    "iv_put_BS": iv_puts,
    "iv_put_approx": iv_puts_approx,
    "rel_err_put": rel_err_puts
})

print(results)


# ## Conclusion
# 
# The Stefanica–Radoicic explicit approximation performs very well overall, especially for put options and for strikes close to the forward price. In the central region of the strike range, the approximate implied volatilities are almost identical to the corresponding Black–Scholes implied volatilities, with relative errors that are very close to zero.
# 
# For put options, the approximation remains accurate across the whole range of strikes considered. The relative error is very small near the money and increases only moderately in the far out-of-the-money region, which suggests that the explicit formula provides a reliable and efficient approximation for implied volatility in this dataset.
# 
# For call options, the approximation is also very accurate for lower and near-the-money strikes, while the error becomes larger for more extreme strikes. Overall, the results show that the explicit formula is highly effective as a fast approximation to Black–Scholes implied volatility, especially in the central region of practical interest.

# In[ ]:




