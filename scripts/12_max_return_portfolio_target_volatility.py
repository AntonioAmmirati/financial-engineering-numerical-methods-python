#!/usr/bin/env python
# coding: utf-8

# ### Exercise 10
# 
# (i) Show that the asset allocation for a maximum return portfolio with variance of return equal to $\sigma_P^2$ can also be written as follows in terms of the asset weights vector $w_T$ of the tangency portfolio:
# 
# - asset weights vector $w_{\max}$ given by
# 
# $$
# w_{\max} =
# \frac{\sigma_P}{\sqrt{w_T^\top \Sigma_R w_T}}
# \cdot
# \operatorname{sign}\!\left(\mathbf{1}^\top \Sigma_R^{-1}\mu\right)\, w_T
# $$
# 
# - weight $w_{\max,\text{cash}}$ of the cash position equal to $1 - \mathbf{1}^\top w_{\max}$, i.e.,
# 
# $$
# w_{\max,\text{cash}} = 1 - \mathbf{1}^\top w_{\max}
# $$
# 
# (ii) Write a pseudocode for computing the asset allocation for a maximum return portfolio with variance of return equal to $\sigma_P^2$ using formulas (3) and (4).

# ### Formal proof of formula (3)
# 
# Let $w$ denote the risky-asset weights vector of a maximum return portfolio with variance equal to $\sigma_P^2$, and let $w_{\text{cash}}$ denote the cash position.
# 
# Since cash is risk-free, it does not contribute to portfolio variance. Therefore, the variance of the portfolio is
# 
# $$
# \operatorname{Var}(R_P) = w^\top \Sigma_R w.
# $$
# 
# A maximum return portfolio with a given variance must lie on the capital allocation line, and therefore its risky-asset allocation must be proportional to the tangency portfolio weights vector $w_T$. Hence, there exists a scalar $\lambda \in \mathbb{R}$ such that
# 
# $$
# w = \lambda w_T.
# $$
# 
# Substituting into the variance expression gives
# 
# $$
# \operatorname{Var}(R_P)
# = (\lambda w_T)^\top \Sigma_R (\lambda w_T)
# = \lambda^2 w_T^\top \Sigma_R w_T.
# $$
# 
# Since the portfolio variance is required to be equal to $\sigma_P^2$, we impose
# 
# $$
# \lambda^2 w_T^\top \Sigma_R w_T = \sigma_P^2.
# $$
# 
# Therefore,
# 
# $$
# \lambda = \pm \frac{\sigma_P}{\sqrt{w_T^\top \Sigma_R w_T}}.
# $$
# 
# To determine which sign gives the maximum return, consider the expected return of the risky part of the portfolio:
# 
# $$
# w^\top \mu = (\lambda w_T)^\top \mu = \lambda\, w_T^\top \mu.
# $$
# 
# Thus, to maximize expected return, $\lambda$ must have the same sign as $w_T^\top \mu$.
# 
# Now, by definition of the tangency portfolio,
# 
# $$
# w_T = \frac{\Sigma_R^{-1}\mu}{\mathbf{1}^\top \Sigma_R^{-1}\mu}.
# $$
# 
# Hence,
# 
# $$
# w_T^\top \mu
# =
# \frac{\mu^\top \Sigma_R^{-1}\mu}{\mathbf{1}^\top \Sigma_R^{-1}\mu}.
# $$
# 
# Since $\Sigma_R$ is positive definite, we have
# 
# $$
# \mu^\top \Sigma_R^{-1}\mu > 0
# $$
# 
# for every nonzero $\mu$. Therefore, the sign of $w_T^\top \mu$ is exactly the sign of
# 
# $$
# \mathbf{1}^\top \Sigma_R^{-1}\mu.
# $$
# 
# It follows that
# 
# $$
# \lambda =
# \frac{\sigma_P}{\sqrt{w_T^\top \Sigma_R w_T}}
# \cdot
# \operatorname{sign}\!\left(\mathbf{1}^\top \Sigma_R^{-1}\mu\right).
# $$
# 
# Substituting this value of $\lambda$ into $w = \lambda w_T$, we obtain
# 
# $$
# w_{\max}
# =
# \frac{\sigma_P}{\sqrt{w_T^\top \Sigma_R w_T}}
# \cdot
# \operatorname{sign}\!\left(\mathbf{1}^\top \Sigma_R^{-1}\mu\right)\, w_T,
# $$
# 
# which is formula (3).
# 
# Finally, since the total portfolio weights, including cash, must sum to $1$, the cash position is
# 
# $$
# w_{\max,\text{cash}} = 1 - \mathbf{1}^\top w_{\max},
# $$
# 
# which is formula (4).

# ### (i)
# 
# To show formula (3), we start from the fact that any maximum return portfolio with a given variance is obtained by combining the tangency portfolio with cash.
# 
# Therefore, the risky-asset allocation must be proportional to the tangency portfolio weights vector $w_T$. We then scale $w_T$ so that the resulting portfolio has variance equal to $\sigma_P^2$, and choose the sign that gives the maximum expected return.
# 
# Finally, once the risky-asset weights are obtained, the cash position is computed so that the total portfolio weights sum to 1.

# ### Step 1: Define the input quantities
# 
# We are given the expected return vector `mu`, the covariance matrix `Sigma` of risky asset returns, and the target standard deviation `sigma_target`.
# 
# Our goal is to compute the maximum return portfolio with variance equal to `sigma_target^2`, using formulas (3) and (4).
# 
# We also define the vector of ones, which will be used in the formulas.

# In[ ]:


import numpy as np

n = 4

mu = np.array([0.06, 0.08, 0.05, 0.10])

vols = np.array([0.15, 0.20, 0.12, 0.25])
corr = np.array([
    [1.00, 0.40, 0.30, 0.20],
    [0.40, 1.00, 0.25, 0.35],
    [0.30, 0.25, 1.00, 0.15],
    [0.20, 0.35, 0.15, 1.00]
])

Sigma = np.diag(vols) @ corr @ np.diag(vols)

sigma_target = 0.12

print("mu =", mu)
print("sigma_target =", sigma_target)
print("Sigma =\n", np.round(Sigma, 4))


# $$
# \Sigma =
# \begin{pmatrix}
# 0.0225 & 0.0120 & 0.0054 & 0.0075 \\
# 0.0120 & 0.0400 & 0.0060 & 0.0175 \\
# 0.0054 & 0.0060 & 0.0144 & 0.0045 \\
# 0.0075 & 0.0175 & 0.0045 & 0.0625
# \end{pmatrix}
# $$

# ### Step 2: Compute the Cholesky decomposition of the covariance matrix
# 
# Since the covariance matrix $\Sigma$ is symmetric positive definite, we can apply Cholesky decomposition and write
# 
# $$
# \Sigma = U^\top U,
# $$
# 
# where $U$ is upper triangular.
# 
# We will use this factorization to solve the linear systems involving $\Sigma$ without computing the inverse explicitly.

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


# ### Step 3: Compute the tangency portfolio direction
# 
# The tangency portfolio is proportional to
# 
# $$
# x = \Sigma^{-1}\mu.
# $$
# 
# Instead of computing the inverse directly, we solve the linear system
# 
# $$
# \Sigma x = \mu.
# $$
# 
# Since $\Sigma = U^\top U$, we solve it in two steps:
# 
# 1. solve
# 
# $$
# U^\top y = \mu
# $$
# 
# using forward substitution;
# 
# 2. solve
# 
# $$
# Ux = y
# $$
# 
# using backward substitution.
# 
# This gives the vector $x$, which is proportional to the tangency portfolio.

# In[ ]:


U = cholesky(Sigma)
y = forward_sub(U.T, mu)
x = backward_sub(U, y)

print("y =", y)
print("x =", x)
print('U: ',U)


# $$
# A = U^\top U
# $$
# 
# $$
# U =
# \begin{pmatrix}
# 0.1500 & 0.0800 & 0.0360 & 0.0500 \\
# 0      & 0.18330303 & 0.01702100 & 0.07364854 \\
# 0      & 0          & 0.11320020 & 0.01277761 \\
# 0      & 0          & 0          & 0.23326514
# \end{pmatrix}
# $$

# ### Step 4: Normalize the tangency portfolio weights
# 
# The tangency portfolio weights are obtained by normalizing the vector $x$ so that the risky-asset weights sum to 1:
# 
# $$
# w_T = \frac{x}{\mathbf{1}^\top x}.
# $$
# 
# This gives the asset weights vector of the tangency portfolio.

# In[ ]:


w_T = x / np.sum(x)

print("w_T =", w_T)
print("sum of tangency weights =", np.sum(w_T))


# ### Step 5: Compute the variance of the tangency portfolio
# 
# We now compute
# 
# $$
# w_T^\top \Sigma w_T,
# $$
# 
# which is the variance of the tangency portfolio.
# 
# This quantity appears in formula (3), because we need to scale the tangency portfolio so that the final portfolio has variance equal to $\sigma_{\text{target}}^2$.

# In[ ]:


var_T = w_T @ Sigma @ w_T
sigma_T = np.sqrt(var_T)

print("var_T =", var_T)
print("sigma_T =", sigma_T)


# ### Step 6: Compute the sign term
# 
# Formula (3) contains the term
# 
# $$
# \operatorname{sign}\!\left(\mathbf{1}^\top \Sigma^{-1}\mu\right).
# $$
# 
# Since we already solved $\Sigma x = \mu$, we know that
# 
# $$
# x = \Sigma^{-1}\mu.
# $$
# 
# Therefore,
# 
# $$
# \mathbf{1}^\top \Sigma^{-1}\mu = \mathbf{1}^\top x,
# $$
# 
# so we can compute the sign term directly from the vector $x$.

# In[ ]:


sign_term = np.sign(np.sum(x))

print("1.T Sigma^{-1} mu =", np.sum(x))
print("sign term =", sign_term)


# ### Step 7: Compute the risky asset allocation of the maximum return portfolio
# 
# Using formula (3), the risky asset allocation is
# 
# $$
# w_{\max} =
# \frac{\sigma_{\text{target}}}{\sqrt{w_T^\top \Sigma w_T}}
# \cdot
# \operatorname{sign}\!\left(\mathbf{1}^\top \Sigma^{-1}\mu\right)\, w_T.
# $$
# 
# So we first compute the scaling factor
# 
# $$
# \frac{\sigma_{\text{target}}}{\sqrt{w_T^\top \Sigma w_T}},
# $$
# 
# and then multiply the tangency portfolio weights by this factor and by the sign term.
# 
# This gives the risky-asset weights of the maximum return portfolio.

# In[ ]:


scale = sigma_target / np.sqrt(var_T)
w_max = scale * sign_term * w_T

print("scale =", scale)
print("w_max =", w_max)
print("sum of risky weights =", np.sum(w_max))


# ### Step 8: Compute the cash position
# 
# Using formula (4), the cash position is
# 
# $$
# w_{\max,\text{cash}} = 1-\mathbf{1}^\top w_{\max}.
# $$
# 
# This ensures that the total portfolio weights, including the risky assets and cash, sum to 1.

# In[ ]:


w_max_cash = 1 - np.sum(w_max)

print("w_max_cash =", w_max_cash)
print("total weight =", np.sum(w_max) + w_max_cash)


# ### Step 9: Verify the target variance
# 
# Finally, we verify that the variance of the resulting portfolio is equal to $\sigma_{\text{target}}^2$.
# 
# Since cash is risk-free, it does not contribute to portfolio variance, so it is enough to check the variance of the risky-asset allocation:
# 
# $$
# w_{\max}^\top \Sigma w_{\max} = \sigma_{\text{target}}^2.
# $$

# In[ ]:


var_p = w_max @ Sigma @ w_max
sigma_p = np.sqrt(var_p)

print("portfolio variance =", var_p)
print("target variance =", sigma_target**2)
print("portfolio standard deviation =", sigma_p)
print("target standard deviation =", sigma_target)


# ### Part (ii)
# 
# We now write the pseudocode for computing the asset allocation of the maximum return portfolio with variance equal to $\sigma_P^2$, using formulas (3) and (4).
# 
# The procedure consists of computing the tangency portfolio, evaluating its variance, scaling it to match the target variance, and then computing the corresponding cash position.

# ### Final pseudocode
# 
# We can summarize the procedure for computing the maximum return portfolio with variance equal to $\sigma_P^2$ as follows.

# # Pseudocode
# 
# # 1. Input mu, Sigma, sigma_target
# # 2. Compute the Cholesky factor U such that Sigma = U.T @ U
# # 3. Solve U.T y = mu with forward substitution
# # 4. Solve U x = y with backward substitution
# # 5. Normalize x to get the tangency portfolio: w_T = x / sum(x)
# # 6. Compute var_T = w_T @ Sigma @ w_T
# # 7. Compute sign_term = sign(sum(x))
# # 8. Compute w_max = (sigma_target / sqrt(var_T)) * sign_term * w_T
# # 9. Compute w_max_cash = 1 - sum(w_max)
# # 10. Return w_max and w_max_cash

# In[ ]:




