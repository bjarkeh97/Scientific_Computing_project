import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
import matplotlib.ticker as ticker





def create_Afuex_2D(sourcefunc, boundary, p):

    # Here I create the T_h and I_h matrices. These have precisely the same form as in the lecture notes. Some manual
    # stuff is done since we are working without elimination of boundary conditions
    h = 1/(2**p)
    N = 1/h
    N = int(N)
    Th = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N-1, N-1)).toarray()
    T_h = np.pad(Th,1)
    T_h[0,0] = h**2
    T_h[N,N] = h**2
    Ih = sp.diags([1],[0], shape=(N-1, N-1)).toarray()
    I_h = np.pad(Ih,1)
    # The final A_h matrix is construced here. Because of the h^2 * I_{N+1} identity matrix in the very top left 
    # corner and bottom right corner I have to change four values manually from zero to 1
    A = (1/(h**2))*np.kron(T_h, I_h) + (1/(h**2))*np.kron(I_h, T_h)
    A[0,0] = 1
    A[N,N] = 1
    A[(N+1)**2-N-1,(N+1)**2-N-1] = 1
    A[(N+1)**2-1,(N+1)**2-1] = 1
    
    
    # A meshgrid is created here on which I will evalute the source function. This vector is the right size for
    # the final result, but it includes every boundary value also, as evaluated through f. This is obviously wrong
    # as these boundary values should be evaluated through b, so that has to be adjusted. I therefore immediately 
    # introduce b1 and b_end as vectors which are the boundary values on the bottom and top of the grid, respectively.
    # f is also reshaped here to be a vector, not an array.
    x,y = np.mgrid[0: 1: complex(0, N+1), 0: 1: complex(0, N+1)]
    x = x.transpose()
    y = y.transpose()

    f = sourcefunc(x,y)
    f = np.reshape(f, (N+1)*(N+1))

    x_axis = np.linspace(0, 1, num = N+1)
    b1 = boundary(x_axis, 0)
    b_end = boundary(x_axis, 1)
    
    # In this section I overwrite the parts of the f vector that represent boundary terms and next-to-boundary terms.
    # In the first loop I overwrite the firts and last parts of f with b1 and b_end, so that the bottom and top of the 
    # 'grid' are boundary values. In the second loop I overwrite values representing the left and right side of the
    # 'grid'. Of course the bottom and left boundaries are just filled with zeros, as sin(xy) is zero when either x
    # or y is zero. In the third loop I overwrite the entries which represent positions next to the right boundary. In
    # the last loop I overwrite the entries which represent positions right below the top boundary. 


    for i in range(0, N+1):
        f[i] = b1[i]
        f[(N+1)*N + i] = b_end[i]

    for i in range(1,N):
        f[i*(N+1)] = 0
        f[i*(N+1)+ N] = boundary(1, i*h)
    
    for i in range(0,N-1):    
        f[2*N+i*(N+1)] = f[2*N+i*(N+1)] + boundary(1, (i+1)*h)/(h**2)
    
    for i in range(0,N-1):     
        f[(N+1)**2-1-2*N+i] = f[(N+1)**2-1-2*N+i] + b_end[i+1]/(h**2)
        
    u_ex_pre = boundary(x,y)
    u_ex = np.reshape(u_ex_pre, (1, (N+1)*(N+1)))
        
    return A , f , u_ex





















#LU factorization

def LUfunc(M):
    A = M.astype(float) #To make sure we have float division
    N = np.shape(A)[0]
    for k in range(N-1):
        if A[k,k] == 0:
            print("Breakdown due to: Zero pivot")
            break
        for i in range(k+1,N):
            A[i,k] = A[i,k]/A[k,k]
            for j in range(k+1,N):
                A[i,j] = A[i,j] - A[i,k]*A[k,j]
    return A

def LUfunc2(A):
    L = np.tril(A,-1)+np.eye(np.shape(A)[0])
    U = np.triu(A)
    return L , U


#Substitution

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




#Final direct solver

def DirSolver(A,f):
    LU = LUfunc(A)
    y = Forsub(LU,f)
    u = Backsub(LU,y)
    return u





