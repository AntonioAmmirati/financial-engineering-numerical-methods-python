#!/usr/bin/env python
# coding: utf-8

# ### Exercise 6
# 
# Assume that an investor allocates wealth between two risky assets and cash. The expected returns of the two risky assets over a three-month period are
# 
# $$
# \mu_1 = 6\%, \qquad \mu_2 = 12\%,
# $$
# 
# with standard deviations
# 
# $$
# \sigma_1 = 15\%, \qquad \sigma_2 = 25\%,
# $$
# 
# and correlation
# 
# $$
# \rho_{12} = 0.30.
# $$
# 
# The risk-free interest rate is
# 
# $$
# r_f = 2\%.
# $$
# 
# The covariance matrix is therefore
# 
# $$
# \Sigma =
# \begin{pmatrix}
# 0.15^2 & 0.30(0.15)(0.25) \\
# 0.30(0.15)(0.25) & 0.25^2
# \end{pmatrix}
# =
# \begin{pmatrix}
# 0.0225 & 0.01125 \\
# 0.01125 & 0.0625
# \end{pmatrix}.
# $$

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


V = 20000000
rf = 0.02
mu_asset = np.array([0.06,0.12])
d_ = np.array([[0.15,0],[0,0.25]])
corr = np.array([[1,0.3],[0.3,1]])
cov = d_ @ corr @ d_


# #### (i) Find the asset allocation for the tangency portfolio.

# Here we want the **risky portfolio with the highest Sharpe ratio**, that is, the portfolio that maximizes excess return per unit of risk.
# 
# So the optimization problem is
# 
# $$
# \max_w \frac{w^\top(\mu - r_f \mathbf{1})}{\sqrt{w^\top \Sigma w}}
# \qquad \text{subject to} \qquad \mathbf{1}^\top w = 1.
# $$
# 
# This is the **tangency portfolio optimization**. Its solution is proportional to
# 
# $$
# x = \Sigma^{-1}(\mu - r_f \mathbf{1}),
# $$
# 
# and then we normalize it to make the risky weights sum to one:
# 
# $$
# w_T = \frac{x}{\mathbf{1}^\top x}.
# $$
# We are solving a **maximum Sharpe ratio problem**.

# In[ ]:


mu_bar = mu_asset - rf


# ### Why we use Cholesky and LU decomposition
# 
# To solve the portfolio optimization problems, we need to solve linear systems involving the covariance matrix $\Sigma$ or block systems coming from first-order conditions of constrained optimization problems.
# 
# Since $\Sigma$ is a covariance matrix, it is symmetric and positive definite, so Cholesky decomposition is appropriate because it is numerically efficient and stable. In particular, we can write
# 
# $$
# \Sigma = U^\top U,
# $$
# 
# where $U$ is an upper triangular matrix, and then solve the system by forward and backward substitution.
# 
# When the optimization problem includes constraints, such as the full-investment condition or a target expected return, we obtain a larger linear system involving both the portfolio weights and Lagrange multipliers. In that case, LU decomposition is convenient because it can be applied to the full block matrix even when the matrix is not symmetric positive definite.
# 
# Thus, Cholesky is used when solving systems directly involving the covariance matrix, while LU is used for the more general constrained linear system arising from the optimization problem.
# 

# In[ ]:


def cholesky(A):
    n = A.shape[0]
    U = np.zeros((n, n))
    
    for i in range(n):
        
       
        U[i, i] = np.sqrt(A[i, i] - np.sum(U[:i, i]**2))
        
        
        for j in range(i+1, n):
            U[i, j] = (A[i, j] - np.sum(U[:i, i] * U[:i, j])) / U[i, i]
    
    return U


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


U = cholesky(cov)
y = forward_sub(U.T,mu_bar)
x = backward_sub(U,y)


# In[ ]:


ones = np.ones(cov.shape[0])
c = 1/(ones@x) 
w_ = c* x


# In[ ]:


print('weights : ',w_)
stock_1 = V * w_[0]
stock_2 = V * w_[1]
print(stock_1, '$')
print(stock_2, '$')


# ##### (ii) Find the asset allocation for a minimum variance portfolio with 8% expected return, and the standard deviation of the return of this portfolio.

# In[ ]:


target_mu = 0.08
c1 = (target_mu - rf)/(mu_bar.T@x)
w_min = c1 * x
print(w_min)
w_cash = (1 - w_min.sum())
print(w_cash)


# In[ ]:


portfolio_var = w_min.T @ cov @ w_min
portfolio_std = np.sqrt(portfolio_var)
print('Volatility :' ,portfolio_std)


# ### Minimum variance portfolio with target return \(8\%\)
# 
# Since a risk-free asset is available, the minimum variance portfolio for a given target return lies on the Capital Allocation Line and is obtained by combining the tangency portfolio with cash.
# 
# Using the tangency direction $x=\Sigma^{-1}(\mu-r_f\mathbf{1})$, and we scale it to match the target return $\mu^*=8\%$:
# 
# $$
# c=\frac{\mu^*-r_f}{(\mu-r_f\mathbf{1})^\top x}
# $$
# 
# The allocation in the two risky assets is then
# 
# $$
# w = c\,x
# $$
# 
# and the cash weight is
# 
# $$
# w_f = 1-\mathbf{1}^\top w
# $$
# 
# The resulting portfolio is:
# 
# $$
# w_1 \approx 0.3511,\qquad
# w_2 \approx 0.4596,\qquad
# w_f \approx 0.1894
# $$
# 
# Thus, about \(35.11\%\) is invested in asset 1, \(45.96\%\) in asset 2, and \(18.94\%\) in the risk-free asset.
# 
# The portfolio standard deviation is
# 
# $$
# \sigma_p = \sqrt{w^\top \Sigma w} \approx 0.1400
# $$
# 
# so the portfolio volatility is approximately **\(14.00\%\)**.
# 
# ------------------------

# #### (iii) Find the asset allocation for a minimum variance portfolio with 15% expected return, and the standard deviation of the return of this portfolio.

# In[ ]:


target_mu_ = 0.15
c2 = (target_mu_ - rf)/(mu_bar.T@x)
w_m= c2 * x
print(w_m)
w_c = (1 - w_m.sum())
print(w_c)


# In[ ]:


p_variance = w_m @ cov @ w_m
p_std = np.sqrt(p_variance)
print('Volatility - 2th min var optimization :',p_std)


# #### Minimum variance portfolio with target return $15\%$
# 
# Using the same tangency-portfolio approach as in part (ii), but setting the target return equal to $15\%$, we obtain the following allocation:
# 
# $$
# w_1 \approx 0.7606,\qquad
# w_2 \approx 0.9957,\qquad
# w_f \approx -0.7564
# $$
# 
# Thus, about $76.06\%$ is invested in asset 1,  $99.57\%$ in asset 2, and $75.64\%$ is borrowed at the risk-free rate.
# 
# The portfolio standard deviation is
# 
# $$
# \sigma_p \approx 0.3034
# $$
# 
# so the portfolio volatility is approximately **$30.34\%$**.
# 
# ------

# ##### (iv) Find the asset allocation for a maximum return portfolio with 20% standard deviation of return, and the expected return of this portfolio.

# In[ ]:


gamma_target = 0.2
c2 = gamma_target/np.sqrt(mu_bar.T @ x)
w_max = c2 * x
print('weights max return optimization : ' , w_max)
print('cash weight : ' , 1 - w_max.sum())


# In[ ]:


#expected return
mu_max = w_max @ mu_asset + (1-w_max.sum())*rf
print('Expected Return :' ,mu_max)


# In[ ]:


print('Check Standard Deviation Target :',np.sqrt(w_max @ cov @ w_max))


# 

# For the maximum return portfolio with standard deviation equal to $30\%$, we use the fact that any maximum return portfolio is obtained by combining the tangency portfolio with the risk-free asset.
# 
# Let $y$ be the proportion invested in the tangency portfolio. Since portfolio volatility scales linearly in $y$, we impose
# 
# $$
# y \, \sigma_T = 0.30.
# $$
# 
# Using the standard deviation of the tangency portfolio, we obtain the corresponding scaling factor and then multiply the tangency portfolio weights by $y$.

# In[ ]:


sigma_target = 0.30

w_max = sigma_target / np.sqrt(mu_bar @ x)
w = w_max * x
wf = 1 - np.sum(w)

mu_p = wf * rf + w @ mu_asset
sigma_p = np.sqrt(w @ cov @ w)


# In[ ]:


print("w1 =", w[0])
print("w2 =", w[1])
print("wf =", wf)
print("mu_p =", mu_p)
print("sigma_p =", sigma_p)


# Thus, the maximum return portfolio with standard deviation equal to $30\%$ is obtained by investing
# 
# $$
# w_1 \approx 0.7522, \qquad w_2 \approx 0.9847, \qquad w_f \approx -0.7369.
# $$
# 
# Hence, the investor allocates approximately $75.22\%$ to asset 1 and $98.47\%$ to asset 2, while borrowing $73.69\%$ at the risk-free rate.
# 
# The expected return of this portfolio is
# 
# $$
# \mu_p \approx 0.1486 = 14.86\%.
# $$
# 
# Its standard deviation is
# 
# $$
# \sigma_p = 0.30 = 30\%.
# $$
# 
# ------

# ##### (vi) Assume that the risk–free interest rate changes to 2.5%. How do you adjust the asset allocation of the minimum variance portfolio with 8% expected return in order to maintain a minimum variance portfolio with 8% expected return? How do you adjust the asset allocation of the maximum return portfolio with 20% standard deviation of return in order to maintain a maximum return portfolio with 20% standard deviation of return?

# In[ ]:


rf_new = 0.025
mu_bar_new = mu_asset - rf_new
y = forward_sub(U.T,mu_bar_new)
x2 = backward_sub(U,y)
c_new = (target_mu - rf_new)/(mu_bar_new.T @ x2)
w_new = c_new * x2
print('New Weights after risk free rate change :', w_new)
print('New Cash Weights :',(1-w_new.sum()))


# In[ ]:


#Maximum Return Optimization Change
c_new_ = gamma_target/np.sqrt(mu_bar_new.T @ x2)
w_new_ = c_new_ * x2
print('Max Return Optimization new weights: ' , w_new_)
print('Cash weights: ',(1-w_new_.sum()))


# ### Effect of the change in the risk-free rate
# 
# When the risk-free rate increases from $2\%$ to $2.5\%$, the excess return vector changes, so both portfolios must be recomputed using the new value of $r_f$.
# 
# For the minimum variance portfolio with $8\%$ expected return, the new allocation is
# 
# $$
# w_1 \approx 0.3004,\qquad
# w_2 \approx 0.4683,\qquad
# w_f \approx 0.2313
# $$
# 
# Thus, the portfolio shifts toward a larger cash position.
# 
# For the maximum return portfolio with $20\%$ standard deviation, the new allocation is
# 
# $$
# w_1 \approx 0.4371,\qquad
# w_2 \approx 0.6812,\qquad
# w_f \approx -0.1183
# $$
# 
# Thus, the portfolio still uses leverage, but less borrowing is needed than before.
# 
# So, to maintain the same portfolio objectives after the change in the risk-free rate, the correct adjustment is to recompute the risky allocation and the cash position using the updated excess return vector.
# 
# ------
