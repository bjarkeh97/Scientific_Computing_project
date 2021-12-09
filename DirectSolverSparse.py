#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import linalg
from scipy import sparse

class SparseMatrixDirectSolver:
    
    #Method of LU decomposition. Returns matrix with L in lower triangular part and U in upper.
    def LUfunc(M):
        Afloat = M.astype(float) #To make sure we have float division
#        A = sparse.csr_matrix(Afloat) #make sparse
        A = Afloat
        N = np.shape(A)[0]    #length of square matrix
        for k in range(N-1):
            if A[k,k] == 0:
                print("Breakdown due to: Zero pivot")
                break
            for i in range(k+1,N):
                A[i,k] = A[i,k]/A[k,k]
                for j in range(k+1,N):
                    A[i,j] = A[i,j] - A[i,k]*A[k,j]
        return A
    
    
    def LUfunc2(A):  #Returns LUfunc matrix result as L and U matrices
        L = np.tril(A,-1)+np.eye(np.shape(A)[0])
        U = np.triu(A)
        return L , U

    #Forward     Works for a LU decomposed matrix from LUfunc      L is NxN matrix, f is N vector
    def Forsub(A,f):
        N = np.shape(A)[0]
        y = np.array([f[0]]) #first element is always just f[0] due to lower tridiagonal
        L = A.astype(float)
        for i in range(N):
            L[i,i] = 1   #Make sure diagonal is 1
        for i in range(1,N):
            y=np.append(y,f[i]-np.dot(L[i,:i],y[:i]))
        return y
    
    #backward    Use y from Forsub
    def Backsub(A,y):
        N = np.shape(A)[0]
        U = A.astype(float)
        u = np.zeros(N)
        u[-1] = y[-1]/U[-1,-1]   #manually add first element to u
        for i in reversed(range(N-1)):
            u[i] = ( y[i]-np.dot(U[i,i:N],u[i:N]) )/U[i,i]
        return u
    
    def DirSolver(A,f):
        LU = LUfunc(A)
        y = Forsub(LU,f)
        u = Backsub(LU,y)
        return u


# In[ ]:




