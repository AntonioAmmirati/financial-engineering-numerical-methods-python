#!/usr/bin/env python
# coding: utf-8

# ### HW1 - ex.11
# The file indices-july2016.csv contains the January 2016 – July 2016 end of day values
# of nine major US indeces.

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('indices-july2016.csv')


# In[ ]:


df.head()


# In[ ]:


df = df.set_index('Date')
df.index = pd.to_datetime(df.index)


# ##### (i) Compute the sample covariance matrix of the daily percentage returns of the indeces, and the corresponding sample corelation matrix. Compute the sample covariance and correlation matrices for daily log returns, and compare them with the corresponding matrices for daily percentage returns.

# In[ ]:


#DAILY PERCENTAGE RETURNS
rets = df.pct_change() 
rets.head()


# In[ ]:


#DAILY LOG RETURNS
log_rets_day = np.log1p(df.pct_change())


# In[ ]:


#CORRELATION MATRIX OF DAILY PERCENTAGE RETURNS
corr_daily = rets.dropna().corr()
#COVARIANCE MATRIX OF DAILY PERCENTAGE RETURNS
cov_daily = rets.dropna().cov()


# In[ ]:


corr_daily


# In[ ]:


cov_daily


# In[ ]:


#DAILY LOG RETURNS
log_rets_day = np.log1p(df.pct_change())
#CORRELATION MATRIX OF LOG DAILY RETURNS
corr_daily_log = log_rets_day.corr()
#COVARIANCE MATRIX OF  LOG DAILY RETURNS
cov_daily_log = log_rets_day.cov()


# In[ ]:


corr_daily_log


# In[ ]:


cov_daily_log


# ------------------------------------------------------------------------------------------
# #### WEEK

# 

# #### (ii) Compute the sample covariance matrix of the weekly percentage returns of the indeces, and the corresponding sample corelation matrix. Compute the sample covariance and correlation matrices for weekly log returns, and compare them with the corresponding matrices for weekly percentage returns.

# In[ ]:


#PREPARING THE DATAFRAME FOR WEEKLY RETURNS
df_week = df.resample('W-FRI').last()
df_week.head()


# In[ ]:


#RETURNS
weekly_rets = df_week.pct_change() 
#CORRELATION MATRIX
corr_weekly = weekly_rets.corr()
#COVARIANCE MATRIX
cov_weekly = weekly_rets.cov()


# In[ ]:


corr_weekly


# In[ ]:


cov_weekly


# In[ ]:


#WEEKLY LOG RETURNS
log_rets_w = np.log1p(df_week.pct_change())
#CORRELATION MATRIX OF LOG DAILY RETURNS
corr_weekly_log = log_rets_w.corr()
#COVARIANCE MATRIX OF  LOG DAILY RETURNS
cov_weekly_log = log_rets_w.cov()


# In[ ]:


corr_weekly_log


# In[ ]:


cov_weekly_log


# ---------------------------------------------------------------------------------------------------------
# ### MONTH

# ##### (iii) Compute the sample covariance matrix of the monthly percentage returns of the indeces, and the corresponding sample corelation matrix. Compute the sample covariance and correlation matrices for monthly log returns, and compare them with the corresponding matrices for monthly percentage returns.

# In[ ]:


#PREPARING THE DATAFRAME FOR MONTHLY RETURNS
df_month = df.resample('ME').last()
df_month.head()


# In[ ]:


#MONTHLY RETURNS
rets_monthly = df_month.pct_change() 
#CORRELATION MATRIX
corr_monthly = rets_monthly.corr()
#COVARIANCE MATRIX
cov_monthly = rets_monthly.cov()


# In[ ]:


corr_monthly


# In[ ]:


cov_monthly


# In[ ]:


#MONTHLY LOG RETURNS
log_rets_m = np.log1p(df_month.pct_change())
#CORRELATION MATRIX OF LOG MONTHLY RETURNS
corr_monthly_log = log_rets_m.corr()
#COVARIANCE MATRIX OF  LOG MONTHLY RETURNS
cov_monthly_log = log_rets_m.cov()


# In[ ]:


corr_monthly_log


# In[ ]:


cov_monthly_log


# ##### (iv) Comment on the diﬀerences between the sample covariance and correlation matrices for daily, weekly, and monthly returns.

# ### Final Comments (daily vs weekly vs monthly; percentage vs log returns)
# 
# - **Covariance depends on scale and horizon.** Since \(\Sigma=\mathrm{Cov}(r)\) is not scale-free, its entries change a lot when we move from **daily → weekly → monthly** (returns aggregate over longer horizons, so variances/covariances typically increase). 
# 
# - **Correlation is scale-free and more comparable.** The correlation matrix \(R\) normalizes by standard deviations, so it is much more stable across (i) different units (percent vs log) and (ii) different magnitudes of volatility. This is why the correlation matrices for **daily % returns** and **daily log returns** are very similar in practice.
# 
# - **Percent vs log returns are close for small returns.** For small \(r\), \(\log(1+r)\approx r\), so **daily** percent and log returns give almost the same dependence structure. Differences can become slightly larger for **weekly/monthly** returns (because returns are larger), but correlations usually remain close.
# 
# - **Effect of sampling frequency on correlations.** Aggregating to weekly/monthly can reduce high-frequency noise and sometimes makes correlations look stronger, but it also leaves **fewer observations**, so weekly/monthly estimates can be less precise (more sampling variability).
# 
# Overall: **covariances change a lot with frequency and return definition**, while **correlations are much more stable and are the best tool to compare dependence across daily/weekly/monthly returns**.
