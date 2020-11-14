# solving linear and quadratic programming using cvxopt

# %% Linear programming example 

import numpy as np 
import cvxopt 


c = np.ones(3)
G = np.array(
    [[-1,0,0],
    [0,-1,0],
    [0,0,-1],
    [-1,1,-1]]
)
h = np.array([2,1,3,-4])
c = cvxopt.matrix(c,tc = 'd')
G = cvxopt.matrix(G,tc = 'd')
h = cvxopt.matrix(h,tc = 'd')

sol = cvxopt.solvers.lp(c,G,h)
xOpt = sol['x']
J = sol['primal objective']
# print(xOpt)
# print(J)

# %% quadratic programs and constrained least-squares exercise 

n= 5
A = np.random.randn(n,n)
b = np.random.randn(n)
l_i = -.5
u_i = .5

P = 2 * A.T @ A 
q = -2 * A.T @ b
# G = np.vstack((np.eye(n),-np.eye(n)))
G = np.concatenate([np.eye(n),-np.eye(n)],axis=0)
h = np.concatenate([u_i*np.ones((n,)),-l_i* np.ones((n,))],axis=0)
# h = np.hstack((u_i*np.ones((n,)),-l_i* np.ones((n,))))
# h = np.vstack((u_i*np.ones((n,1)),-l_i* np.ones((n,1))))
#### 1D array can be (n,1) or (1,n) ######

P = cvxopt.matrix(P, tc='d')
q = cvxopt.matrix(q, tc='d')
G = cvxopt.matrix(G, tc='d')
h = cvxopt.matrix(h, tc='d')
sol = cvxopt.solvers.qp(P,q,G,h)

# print('x*=', sol['x'])
# print('p*=', sol['primal objective'] + b.T@b)


# %% exercises 

# n = 2
# c = np.array([3,2])
# G = -np.eye(2)
# h = np.array([0,0]).reshape(n,1)


# n = 2
# c = np.array([1,0])
# G = -np.eye(2)
# h = np.array([0,0]).reshape(n,1)



# c = np.array([-5,-7])
# G = np.array([
#     [-3,-2],
#     [-2,1],
#     [-1,0],
#     [0,-1]
# ])
# h = np.array([30,12,0,0]).reshape(4,1)

c = np.array([3,1])
G = np.array([
    [1,-1],
    [3,2],
    [2,3],
    [2,-3],
    [0,-1],
    [-1,0]
])
h = np.array([1,12,3,-9,0,0])

c = cvxopt.matrix(c, tc='d')
G = cvxopt.matrix(G, tc='d')
h = cvxopt.matrix(h, tc='d')

sol = cvxopt.solvers.lp(c,G,h)
xOpt = sol['x']
J = sol['primal objective']
# print(xOpt)
# print(J)

# %% quadratic exercises

# P = 2 * np.eye(2)
# q = np.zeros((2,1))
# G = -np.eye(2)
# h = np.array([-1,-1]).reshape((2,1))

# P = 2 * np.diag([2,7])
# q = np.zeros((2,1))
# G = np.diag([-1,1])
# h = np.array([3,2])

# P = 2 * np.eye(2)
# q = np.zeros((2,1))
# G = np.array([
#     [1,0],
#     [0,1],
#     [4,3]
# ])
# h = np.array([-3,4,0])


# P = cvxopt.matrix(P,tc = 'd')
# q = cvxopt.matrix(q,tc = 'd')
# G = cvxopt.matrix(G,tc = 'd')
# h = cvxopt.matrix(h,tc = 'd')

# sol = cvxopt.solvers.qp(P,q,G,h)
# print('x* = ',sol['x'])
# print('J* = ',sol['primal objective'])

P = np.diag([1,1,0.1])
q = np.array([0,0,.55])
G = -np.eye(3)
h = np.zeros((3,1))
A = np.ones((1,3))
b = np.ones(1)

P = cvxopt.matrix(P,tc = 'd')
q = cvxopt.matrix(q,tc = 'd')
G = cvxopt.matrix(G,tc = 'd')
h = cvxopt.matrix(h,tc = 'd')
A = cvxopt.matrix(A,tc = 'd')
b = cvxopt.matrix(b,tc = 'd')

sol = cvxopt.solvers.qp(P,q,G,h,A,b)
print('x* = ',sol['x'])
print('J* = ',sol['primal objective'])
