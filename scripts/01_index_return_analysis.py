#!/usr/bin/env python
# coding: utf-8

# ### HW1 ex.10
# The file indeces-close-jan3-jan31-2017.xlsx contains the January 3, 2017 – January 31,
# 2017 end of day values of Dow Jones, Nasdaq, and S&P 500.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_excel('indeces-close-jan3-jan31-2017.xlsx')


# In[ ]:


df.head()


# In[ ]:


df = df.set_index('Date')


# ##### (i) Compute the log daily returns of the three indices over the given time period.

# In[ ]:


#CALCULATING DAILY LOG-RETURNS
log_rets = np.log1p(df.pct_change())
log_rets.head(5)


# ##### (ii) Compute the sample covariance matrix of the log daily returns of the three indices over the given time period.

# In[ ]:


#SAMPLE COVARIANCE OF LOG RETURNS
sample_cov_log = log_rets.dropna().cov()


# In[ ]:


sample_cov_log


# ##### (iii) Compute the percentage daily returns of the three indices over the given time period.

# In[ ]:


#PERCENTAGE RETURNS
rets = df.pct_change()
rets.head()


# ##### (iv) Compute the sample covariance matrix of the percentage daily returns of the three indices over the given time period.

# In[ ]:


sample_cov = rets.dropna().cov()
sample_cov


# #### RESULTS (SUMMARY)
# ##### The covariance matrices computed from daily simple returns and daily log returns are nearly identical, since for small daily returns we have log(1+r) ≈ r. The NASDAQ shows the largest variance (highest volatility) and all pairwise covariances are positive, indicating that the indices generally move together.
