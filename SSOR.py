#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import linalg
from scipy import sparse

class SparseMatrixDirectSolver:
    
    def SORit(self,A,u,w,f):
        N = np.size(u)
        s0 = u[0]
        u[0] = (f[0] - np.dot(A[0,1:],u[1:]))/A[0,0]
        u[0] = (1-w)*s0+w*u[0]
        for i in range(1,N-1):
            s = u[i]
            u[i] = (f[i]-np.dot(A[i,:i-1],u[:i-1])-np.dot(A[i,i+1:],u[i+1:]))/A[i,i]
            u[i] = (1-w)*s+w*u[i]
        return u

    def SOR(self,A,u,w,f,N):
        u0 = u
        for i in range(N):
            print(u0)
            u0 = SORit(A,u0,w,f)
        return u0

    
    
    # In[ ]:




