#!/usr/bin/env python
# coding: utf-8

# #### (4) Two stocks trade at $80  and  $65, respectively. Their three-months returns have ex-pected values of 10% and 5%, respectively, and standard deviation of 35% and 15%,respectively. The correlation of the returns is 15%.

# In[ ]:


import pandas as pd
import numpy as np
import yfinance as yf


# ##### (i) Consider a portfolio made of 150 shares of the first stock and 500 shares of the second stock. What are the weights of each stock in this portfolio?

# In[ ]:


d_ = np.array([[0.35,0],[0,0.15]])
corr = np.array([[1,0.15],[0.15,1]])
cov = d_ @ corr @ d_


# In[ ]:


vals = np.array([ (80*150) , (500*65)])
vals = vals/ vals.sum()
print(vals)
print(vals.sum())


# ##### (ii) Assume that you have $1,000,000 to invest. Find a portfolio made of the two stocks that has 8% expected return.

# In[ ]:


# 0.08 = w*0.1 + (1-w)*0.05 , 0.08 = 0.1w + 0.05 - 0.05w , 0.03 = 0.05w , w = 0.03/0.05 = 0.6 , w2 = 1 - 0.6 = 0.4
w1 = 0.6
w2 = 0.4
stock_1 = 1000000 * 0.6
stock_2 = 1000000 * 0.4
print(stock_1,'$')
print(stock_2,'$')


# ##### (iii) Identify the two portfolios fully invested in the two assets that have a 24% standard deviation of return. What are the expected returns of the two portfolios?

# #### Portfolios with $24\%$ standard deviation
# 
# We use the two-asset portfolio variance formula:
# 
# $$
# \sigma_p^2
# =
# w_1^2 \sigma_1^2
# +
# w_2^2 \sigma_2^2
# +
# 2 w_1 w_2 \rho \sigma_1 \sigma_2
# $$
# 
# Since the portfolio is fully invested,
# 
# $$
# w_1 + w_2 = 1
# \quad \Longrightarrow \quad
# w_2 = 1 - w_1
# $$
# 
# We are given:
# 
# $$
# \sigma_p = 0.24,\qquad \sigma_1 = 0.35,\qquad \sigma_2 = 0.15,\qquad \rho = 0.15
# $$
# 
# Substituting into the variance formula gives
# 
# $$
# 0.24^2
# =
# w_1^2 (0.35)^2
# +
# (1-w_1)^2 (0.15)^2
# +
# 2w_1(1-w_1)(0.15)(0.35)(0.15)
# $$
# 
# that is,
# 
# $$
# 0.0576
# =
# 0.1225\,w_1^2
# +
# 0.0225(1-w_1)^2
# +
# 0.01575\,w_1(1-w_1)
# $$
# 
# This is a quadratic equation in $w_1$. Once the two solutions for $w_1$ are found, the corresponding weights of the second asset are
# 
# $$
# w_2 = 1 - w_1
# $$
# 
# Finally, the expected return of each portfolio is computed using
# 
# $$
# \mu_p = w_1 \mu_1 + w_2 \mu_2
# $$
# 
# with
# 
# $$
# \mu_1 = 0.10,\qquad \mu_2 = 0.05
# $$

# In[ ]:




