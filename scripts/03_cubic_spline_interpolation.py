#!/usr/bin/env python
# coding: utf-8

# #### HW3 - ex 2 - Cubic Spline Interpolation 

# The following discount factors are obtained by fitting market data
# The overnight rate is 1%.
# 

# (i) What is the linear system that has to be solved for the cubic spline interpolation
# of the zero rate curve?

# #### First we use the discount factor and the time t to calculate the discount rate

# In[ ]:


d1 = 0.9980
d2 = 0.9935
d3 = 0.9820
d4 = 0.9775

t1 = 2/12
t2 = 5/12
t3 = 11/12
t4 = 15/12
#we already have the overnight rate that would be r(0,0)
r0 = 0.01


# In[ ]:


import numpy as np
r1 = (-np.log(d1))/t1
r2 = (-np.log(d2))/t2
r3 = (-np.log(d3))/t3
r4 = (-np.log(d4))/t4


# the linear system that has to be solved is the follow:
# for each interval -> [0,2/12] ; [2/12,5/12] ; [5/12,11/12] ; [11/12,15/12]
# 
# 1) a(i-1) + b(i-1)x + c(i-1)x^2 + d(i-1) x^3 = v(i-1) --> f(i-1) = r0
# 2) a(i) + b(i)x + c(i)x^2 + d(i)x^3 = v(i) --> f(i) = r1
# then for every middle point , we need to assure that the first derivative from the left and right is the same
# f'(i) = f'(i+1)
# 3) b(i) + 2c(i)x + 3d(i)x^2 = b(i+1) + 2c(i+1)x + 3d(i+1)x^2
# then to assure the smoothess of our interpolation we need to make sure that the second derivative exists, so:
# f''(i) = f''(i+1)
# 4) 2c(i) + 6d(i)x = 2c(i+1) + 6d(i)x
# and in the end since we follow the natural spline approach, we set as the last two conditions:
# f''(x0) = 0
# f''(xn) = 0
# 5) 2c(i0)= 0
# 6) 2c(n) + 6d(n)x = 0
# So our functions in the start and in the end we behave as a linear.

# In[ ]:


#now we construct the linear system


# In[ ]:


b = np.array([0,r0,r1,r1,r2,r2,r3,r3,r4,0,0,0,0,0,0,0])


# In[ ]:


print(b)


# #### x = np.array([a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3,a4,b4,c4,d4])

# In[ ]:


#Matrix M
M =np.array([[0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [1,t1,t1**2,t1**3,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,t1,t1**2,t1**3,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,t2,t2**2,t2**3,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,t2,t2**2,t2**3,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,t3,t3**2,t3**3,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,t3,t3**2,t3**3],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,t4,t4**2,t4**3],
            [0,1,2*t1,3*t1**2,0,-1,-2*t1,-3*t1**2,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,2*t2,3*t2**2,0,-1,-2*t2,-3*t2**2,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,2*t3,3*t3**2,0,-1,-2*t3,-3*t3**2],
            [0,0,2,6*t1,0,0,-2,-6*t1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,2,6*t2,0,0,-2,-6*t2,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,2,6*t3,0,0,-2,-6*t3],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,6*t4]])


# In[ ]:


print(M)


# In[ ]:


x = np.linalg.solve(M, b)


# In[ ]:


x


# In[ ]:


coeffs = x.reshape(4,4)   
a1,b1,c1,d1 = coeffs[0]
a2,b2,c2,d2 = coeffs[1]
a3,b3,c3,d3 = coeffs[2]
a4,b4,c4,d4 = coeffs[3]


# In[ ]:


def z1(t): return a1 + b1*t + c1*t**2 + d1*t**3   # t in [t0,t1]
def z2(t): return a2 + b2*t + c2*t**2 + d2*t**3   # t in [t1,t2]
def z3(t): return a3 + b3*t + c3*t**2 + d3*t**3   # t in [t2,t3]
def z4(t): return a4 + b4*t + c4*t**2 + d4*t**3   # t in [t3,t4]


# In[ ]:


z1(t1)


# In[ ]:


z2(t2)


# In[ ]:


z3(t3)


# In[ ]:


z4(t4)


# In[ ]:


print(z1(t1), z2(t1))
print(z2(t2), z3(t2))
print(z3(t3), z4(t3))


# (iii) Find the value of a 13 months quarterly bond with 2.5% coupon rate.

# In[ ]:


F = 100
c = (0.025/4) * 100
t = [1/12,4/12,7/12,10/12]
T = 13/12
final = F + c


# In[ ]:


def B0(T):
    return c*(np.exp(-z1(t[0])*t[0])+np.exp(-z2(t[1])*t[1]) + np.exp(-z3(t[2])*t[2]) +  
                                           np.exp(-z3(t[3])*t[3]))+ final* np.exp(-z4(T)*T)


# In[ ]:


B0(T)


# In[ ]:




