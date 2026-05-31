#!/usr/bin/env python
# coding: utf-8

# ## Exercise 8
# 
# Consider four assets with the following expected returns over a fixed time period:
# 
# $$
# \mu_1 = 5.1\%, \quad \mu_2 = 4.5\%, \quad \mu_3 = 6.8\%, \quad \mu_4 = 4.2\%
# $$
# 
# and with the following covariance matrix of their returns over the same time period:
# 
# $$
# \Sigma =
# \begin{pmatrix}
# 0.09 & -0.01 & -0.03 & -0.02 \\
# -0.01 & 0.0625 & 0.02 & -0.01 \\
# -0.03 & 0.02 & 0.1225 & -0.015 \\
# -0.02 & -0.01 & -0.015 & 0.0576
# \end{pmatrix}.
# $$
# 
# Assume that the risk-free interest rate is
# 
# $$
# r_f = 1.5\%.
# $$

# In[ ]:


#and with the following covariance matrix of their returns over the same time period:
cov = np.array([[0.09 , -0.01,-0.03,-0.02],[-0.01,0.0625,0.02,-0.01],[-0.03, 0.02 ,0.1225,-0.015],[-0.02,-0.01,-0.015,0.0576]])
#Assume that the risk–free interest rate is 1.5%.


# #### (i)
# Find the asset allocation for the **tangency portfolio**.  
# Find the expected return and the standard deviation of the return of the tangency portfolio.  
# What is the **Sharpe ratio** of the tangency portfolio?

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


print(cov)


# $$
# \mathrm{cov} =
# \begin{pmatrix}
# 0.09 & -0.01 & -0.03 & -0.02 \\
# -0.01 & 0.0625 & 0.02 & -0.01 \\
# -0.03 & 0.02 & 0.1225 & -0.015 \\
# -0.02 & -0.01 & -0.015 & 0.0576
# \end{pmatrix}
# $$

# In[ ]:


rf = 0.015
mu_asset = np.array([0.051,0.045,0.068,0.042])
mu_bar = mu_asset - rf
print(mu_bar)


# ### Tangency portfolio setup
# 
# To find the tangency portfolio, we use the standard result that the portfolio of risky assets with the highest Sharpe ratio has weights proportional to
# 
# $$
# \Sigma^{-1}(\mu-r_f\mathbf{1})
# $$
# 
# where:
# 
# - $\mu$ is the vector of expected returns,
# - $r_f$ is the risk-free rate,
# - $\mathbf{1}$ is the vector of ones,
# - and $\Sigma$ is the covariance matrix of asset returns.
# 
# Thus, the tangency portfolio weights are given by
# 
# $$
# w_T
# =
# \frac{\Sigma^{-1}(\mu-r_f\mathbf{1})}{\mathbf{1}^\top \Sigma^{-1}(\mu-r_f\mathbf{1})}
# $$
# 
# so that the weights sum to $1$.
# 
# In this problem,
# 
# $$
# \mu =
# \begin{bmatrix}
# 0.051\\
# 0.045\\
# 0.068\\
# 0.042
# \end{bmatrix},
# \qquad
# r_f = 0.015
# $$
# 
# hence the vector of excess expected returns is
# 
# $$
# \mu-r_f\mathbf{1}
# =
# \begin{bmatrix}
# 0.036\\
# 0.030\\
# 0.053\\
# 0.027
# \end{bmatrix}.
# $$
# 
# To compute the tangency portfolio, we need to solve the linear system
# 
# $$
# \Sigma x = \mu-r_f\mathbf{1}.
# $$
# 
# Instead of computing $\Sigma^{-1}$ directly, we use **Cholesky decomposition**, since the covariance matrix is symmetric positive definite. This allows us to write
# 
# $$
# \Sigma = U^\top U
# $$
# 
# with $U$ upper triangular, and then solve the system in two steps:
# 
# $$
# U^\top y = \mu-r_f\mathbf{1},
# \qquad
# Ux = y.
# $$
# 
# We use forward substitution for the first system and backward substitution for the second. This is numerically more stable and more efficient than computing the inverse explicitly.
# 
# Finally, once $x$ is obtained, we normalize it so that the portfolio is fully invested.

# In[ ]:


#we start with cholesky decomposition
def cholesky(A):
    n = A.shape[0]
    U = np.zeros((n,n))
    for i in range(n):
        U[i,i] = np.sqrt(A[i,i] - np.sum(U[:i,i]**2))
        for j in range(i+1,n):
            U[i,j] = (A[i,j] - np.sum(U[:i,i] * U[:i,j]))/U[i,i]
    return U
#now to solve the linear system we do forward and the backward substitution
def forward_sub(L,b):
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i,j]*y[j]
        y[i] = (b[i] - s)/L[i,i]
    return y

def backward_sub(U,y):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        s = 0.0
        for j in range(i+1,n):
            s += U[i,j]*x[j]
        x[i] = (y[i] -s)/U[i,i]
    return x


# In[ ]:


U = cholesky(cov)
y = forward_sub(U.T,mu_bar)
x = backward_sub(U,y)


# In[ ]:


#Tangency portfolio
ones = np.ones(len(cov))
c = 1/(ones @ x)
w_tangency = c * x
print('Weights: ',w_tangency)
print(w_tangency.sum())


# In[ ]:


mu_portfolio = w_tangency @ mu_asset
print('Expected Return :', mu_portfolio)


# In[ ]:


portfolio_variance = w_tangency @ cov @ w_tangency
portfolio_std = np.sqrt(portfolio_variance)
print('Portfolio Standard Deviation :',portfolio_std)


# In[ ]:


sharpe_ratio = (mu_portfolio -rf)/portfolio_std
print('Sharpe Ratio: ',sharpe_ratio)


# ### Tangency portfolio results
# 
# After solving for the tangency portfolio, we obtain the following asset allocation:
# 
# $$
# w_T \approx
# \begin{bmatrix}
# 0.2846 \\
# 0.1757 \\
# 0.2125 \\
# 0.3271
# \end{bmatrix}
# $$
# 
# Thus, the portfolio invests approximately:
# 
# - $28.46\%$ in asset 1,
# - $17.57\%$ in asset 2,
# - $21.25\%$ in asset 3,
# - $32.71\%$ in asset 4.
# 
# The expected return of the tangency portfolio is
# 
# $$
# \mu_T = w_T^\top \mu \approx 0.0506
# $$
# 
# so the portfolio expected return is approximately **$5.06\%$**.
# 
# The portfolio standard deviation is
# 
# $$
# \sigma_T = \sqrt{w_T^\top \Sigma w_T} \approx 0.1040
# $$
# 
# so the portfolio volatility is approximately **$10.40\%$**.
# 
# Finally, the Sharpe ratio is
# 
# $$
# \frac{\mu_T-r_f}{\sigma_T}
# =
# \frac{0.0506-0.015}{0.1040}
# \approx 0.3423
# $$
# 
# This means that the tangency portfolio delivers about **0.342 units of excess return per unit of risk**, which is the highest achievable Sharpe ratio among all fully invested portfolios of the four risky assets.
# 
# -----

# #### (ii)
# Find the asset allocation for a **minimum variance portfolio** with expected return equal to $5\%$, and the standard deviation of the return of this portfolio.  
# What is the **Sharpe ratio** of this portfolio?

# In[ ]:


mu_target = 0.05
c1 = (mu_target - rf)/(mu_bar @ x)
w_min = c1 * x
w_cash = 1 - w_min.sum()
print('Weights :',w_min)
print('Cash Weights :', w_cash)


# In[ ]:


expected_return = w_min @ mu_asset + w_cash * rf
portfolio_variance = w_min @ cov @ w_min
portfolio_std = np.sqrt(portfolio_variance)
sharpe_ratio_ = (expected_return - rf)/portfolio_std
print('Expected Return Portfolio :' , expected_return)
print('Portfolio Volatility :', portfolio_std)
print('Sharpe Ratio :', sharpe_ratio_)


# ### Minimum variance portfolio with $5\%$ expected return
# 
# Using the tangency-portfolio direction and combining it with the risk-free asset, we obtain the minimum variance portfolio that achieves the target expected return of $5\%$.
# 
# The resulting allocation is
# 
# $$
# w \approx
# \begin{bmatrix}
# 0.2797 \\
# 0.1727 \\
# 0.2089 \\
# 0.3215
# \end{bmatrix},
# \qquad
# w_f \approx 0.0173
# $$
# 
# Thus, the portfolio invests about:
# 
# - $27.97\%$ in asset 1,
# - $17.27\%$ in asset 2,
# - $20.89\%$ in asset 3,
# - $32.15\%$ in asset 4,
# - and $1.73\%$ in the risk-free asset.
# 
# The expected return is exactly
# 
# $$
# \mu_p = 0.05
# $$
# 
# and the portfolio standard deviation is
# 
# $$
# \sigma_p \approx 0.1022
# $$
# 
# so the portfolio volatility is approximately **$10.22\%$**.
# 
# The Sharpe ratio is
# 
# $$
# \frac{\mu_p-r_f}{\sigma_p} \approx 0.3423
# $$
# 
# which is the same as the Sharpe ratio of the tangency portfolio, as expected for any portfolio lying on the Capital Allocation Line.
# 
# ------

# #### (iii)
# Find the asset allocation for a **maximum return portfolio** with standard deviation of return equal to $29\%$, and the expected return of this portfolio.  
# What is the **Sharpe ratio** of this portfolio?

# In[ ]:


gamma_target = 0.29
c2 = gamma_target/np.sqrt(mu_bar @ x)
w_max = c2 * x
cash_weight = 1 - w_max.sum()
print('Maximum Return Optimization Weights :',w_max)
print('Cash Weight :',cash_weight)


# In[ ]:


expected_ret = w_max @ mu_asset + rf * cash_weight
variance = w_max @ cov @ w_max
std = np.sqrt(variance)
sharpe_ratio_max_ret = (expected_ret - rf)/std
print('Expected Return Max Return Optimization :',expected_ret)
print('Check Target Standard Deviation :' , std)
print('Sharpe Ratio Max Return Portfolio :',sharpe_ratio_max_ret)


# ### Maximum return portfolio with $29\%$ standard deviation
# 
# Using the tangency-portfolio direction and scaling it to achieve a target volatility of $29\%$, we obtain the following allocation:
# 
# $$
# w \approx
# \begin{bmatrix}
# 0.7934 \\
# 0.4899 \\
# 0.5925 \\
# 0.9119
# \end{bmatrix},
# \qquad
# w_f \approx -1.7876
# $$
# 
# Thus, the portfolio invests about:
# 
# - $79.34\%$ in asset 1,
# - $48.99\%$ in asset 2,
# - $59.25\%$ in asset 3,
# - $91.19\%$ in asset 4,
# 
# and borrows about $178.76\%$ at the risk-free rate.
# 
# The expected return of this portfolio is
# 
# $$
# \mu_p \approx 0.1143
# $$
# 
# so the portfolio expected return is approximately **$11.43\%$**.
# 
# A direct check confirms that the portfolio standard deviation is exactly **$29\%$**.
# 
# The Sharpe ratio is
# 
# $$
# \frac{\mu_p-r_f}{\sigma_p} \approx 0.3423
# $$
# 
# which is the same as the Sharpe ratio of the tangency portfolio, as expected for any portfolio lying on the Capital Allocation Line.
# 
# -----

# #### (iv)
# Find the asset allocation for the **minimum variance portfolio fully invested in the assets** (that is, with no cash position).  
# What is the **Sharpe ratio** of this portfolio?

# ### Fully invested minimum variance portfolio
# 
# To find the minimum variance portfolio fully invested in the risky assets, we minimize the portfolio variance subject to the constraint that the portfolio weights sum to one:
# 
# $$
# \min_w \; w^\top \Sigma w
# \qquad \text{subject to} \qquad
# \mathbf{1}^\top w = 1
# $$
# 
# The solution is the global minimum variance portfolio:
# 
# $$
# w_{GMV} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1}\mathbf{1}}
# $$
# 
# where $\mathbf{1}$ is the vector of ones.
# 
# To compute this, we solve the linear system
# 
# $$
# \Sigma x = \mathbf{1}
# $$
# 
# and then normalize the solution so that the portfolio is fully invested.
# 
# After obtaining the weights, we compute:
# 
# - the expected return
# $$
# \mu_{GMV} = w_{GMV}^\top \mu
# $$
# 
# - the standard deviation
# $$
# \sigma_{GMV} = \sqrt{w_{GMV}^\top \Sigma w_{GMV}}
# $$
# 
# - and the Sharpe ratio
# $$
# \text{Sharpe Ratio} = \frac{\mu_{GMV} - r_f}{\sigma_{GMV}}
# $$
# 
# Even though this portfolio is constructed without a cash position, its Sharpe ratio is still measured relative to the risk-free rate.

# In[ ]:


y = forward_sub(U.T , ones)
x = backward_sub(U,y)
w_gmv = x /(ones.T @ x)
print('Weights Gmv :',w_gmv)


# In[ ]:


expected_gmv = w_gmv.T @ mu_asset
vol = np.sqrt(w_gmv.T @ cov @ w_gmv)
sharpe_ratio = (expected_gmv - rf)/vol
print('Expected Return Global Minimum Variance :', expected_gmv)
print('Standard Deviation Global Minimum Variance :',vol)
print('Sharpe Ratio Global Minimum Variance :',sharpe_ratio)


# ### Global minimum variance portfolio
# 
# The fully invested minimum variance portfolio is obtained by minimizing portfolio variance subject to the constraint that the portfolio weights sum to one.
# 
# The resulting allocation is
# 
# $$
# w_{GMV} \approx
# \begin{bmatrix}
# 0.2713 \\
# 0.2154 \\
# 0.1595 \\
# 0.3538
# \end{bmatrix}
# $$
# 
# Thus, the portfolio invests about:
# 
# - $27.13\%$ in asset 1,
# - $21.54\%$ in asset 2,
# - $15.95\%$ in asset 3,
# - $35.38\%$ in asset 4.
# 
# The expected return of the global minimum variance portfolio is
# 
# $$
# \mu_{GMV} \approx 0.0492
# $$
# 
# so the portfolio expected return is approximately **$4.92\%$**.
# 
# Its standard deviation is
# 
# $$
# \sigma_{GMV} \approx 0.1020
# $$
# 
# so the portfolio volatility is approximately **$10.20\%$**.
# 
# The Sharpe ratio is
# 
# $$
# \frac{\mu_{GMV}-r_f}{\sigma_{GMV}} \approx 0.3357
# $$
# 
# which is lower than the Sharpe ratio of the tangency portfolio, as expected.
