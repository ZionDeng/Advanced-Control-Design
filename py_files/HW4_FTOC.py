# Finite Time Optimal Control 

# %% vehicle 

import matplotlib.pyplot as plt 
import numpy as np 
import pyomo.environ as pyo 

Ts = .2 
N = 70
TFinal = Ts * N

z0Bar = np.array([0,3,0,0])
zNBar = np.array([0,0,0,-np.pi/2])
zMax = np.array([20,10,10,2*np.pi])
zMin = np.array([-20,-.2,-10,-2*np.pi])
umin = [-0.3,-0.6]
umax = [0.3,0.6]

nz = 4
nu = 2 
l_r = 1.738 

m = pyo.ConcreteModel()
m.tidx = pyo.Set(initialize = range(N+1))
m.zidx = pyo.Set(initialize = range(nz))
m.uidx = pyo.Set(initialize = range(nu))

m.z = pyo.Var(m.zidx,m.tidx)
m.u = pyo.Var(m.uidx,m.tidx)

m.cost = pyo.Objective(
    expr = sum((m.z[i,t] - zNBar[i])**2 for i in m.zidx for t in m.tidx 
    if t > N-3 and t<N+1),
    sense = pyo.minimize
)

m.c11 = pyo.Constraint(
    m.tidx, rule = lambda m,t:
    m.z[0,t+1] == m.z[0,t] + Ts * (m.z[2,t]* pyo.cos(m.z[3,t]+m.u[1,t]))
    if t < N else pyo.Constraint.Skip
)
m.c12 = pyo.Constraint(
    m.tidx, rule = lambda m,t:
    m.z[1,t+1] == m.z[1,t] + Ts * (pyo.sin(m.z[3,t]+m.u[1,t]))
    if t < N else pyo.Constraint.Skip
)
m.c13 = pyo.Constraint(
    m.tidx, rule = lambda m,t:
    m.z[2,t+1] == m.z[2,t] +Ts* m.u[0,t]
    if t < N else pyo.Constraint.Skip
) 
m.c14 = pyo.Constraint(
    m.tidx, rule = lambda m,t:
    m.z[3,t+1] == m.z[3,t] + Ts * (m.z[2,t]/l_r * pyo.sin(m.u[1,t]))
    if t < N else pyo.Constraint.Skip
)
# zmin <= zk <= zmax 
m.c21 = pyo.Constraint(
    m.zidx,m.tidx, rule = lambda m,i,t:
    m.z[i,t] <= zMax[i]
    if t< N else pyo.Constraint.Skip
)
m.c22 = pyo.Constraint(
    m.zidx,m.tidx, rule = lambda m,i,t:
    m.z[i,t] >= zMin[i]
    if t< N else pyo.Constraint.Skip
)
# umin <= uk <= umax
m.c31 = pyo.Constraint(
    m.uidx, m.tidx, rule = lambda m,i,t:
    m.u[i,t] <= umax[i]
    if t <N else pyo.Constraint.Skip
)
m.c32 = pyo.Constraint(
    m.uidx, m.tidx, rule = lambda m,i,t:
    m.u[i,t] >= umin[i]
    if t <N else pyo.Constraint.Skip
)
# |beta_k+1 - beta_k| <= beta_d
m.c41 = pyo.Constraint(
    m.tidx, rule = lambda m,t:
    m.u[1,t+1] - m.u[1,t] <= 0.2 
    if t < N-1 else pyo.Constraint.Skip
)
m.c41 = pyo.Constraint(
    m.tidx, rule = lambda m,t:
    m.u[1,t+1] - m.u[1,t] >= -0.2 
    if t < N-1 else pyo.Constraint.Skip
)

m.c5 = pyo.Constraint(
    m.zidx, rule = lambda m,i:
    m.z[i,0] == z0Bar[i]
)
m.c6 = pyo.Constraint(
    m.zidx, rule = lambda m,i:
    m.z[i,N] == zNBar[i]
)

# results = pyo.SolverFactory('ipopt').solve(m).write()


import numpy as np 
from scipy.linalg import block_diag
from numpy.linalg import inv 

def Sx_Su(A,B,N):

    nX = np.size(A,0)
    nU = np.size(B,1)
    Sx = np.eye(nX)

    A_tmp = A
    for i in range(N):
        Sx = np.vstack((Sx,A_tmp))
        A_tmp = A_tmp @ A 

    SxB = Sx @ B 
    Su = np.zeros((nX*(N+1),nU * N))
    for j in range(N):
        Su_tmp = np.vstack((np.zeros((nX,nU)),SxB[:-nX,:]))
        Su[:,j] = Su_tmp.reshape(Su_tmp.shape[0],)
        SxB = Su_tmp
    
    return Sx, Su 

def lqrBatch(A,B,Q,R,PN,N):
    Sx, Su = Sx_Su(A,B,N)
    Qbar =  block_diag(np.kron(np.eye(N),Q),PN)
    Rbar = np.kron(np.eye(N),R)
    QSu = Qbar @ Su
    H = Su.T @ QSu + Rbar
    F = Sx.T @ QSu
    K = -inv(H) @ F.T 
    P0 = F@K + Sx.T @ Qbar @ Sx

    return K,P0

A = np.array([.77, -0.35, 0.49, 0.91]).reshape((2,2))
B = np.array([0.04,0.15]).reshape((2,1))
Q = np.diag([500,100])
R = 1 
PN = np.diag([1500,100])
x0 = np.array([1,-1]).T
N = 5

K,P0 = lqrBatch(A,B,Q,R,PN,N)
U0_star = K @ x0
J0_star = x0.T @ P0 @ x0

# print('u0* = ',U0_star)
# print('J0* = ',J0_star)


Qbar = block_diag(np.kron(np.eye(N),Q),PN)
Rbar = np.kron(np.eye(N),R)
Sx, Su = Sx_Su(A,B,N)
QSu = Qbar @ Su
H = Su.T @ QSu +Rbar
F = Sx.T @ QSu

P = 2*H 
q = 2 * x0.T @ F 

import cvxopt

P = cvxopt.matrix(P,tc= 'd')
q = cvxopt.matrix(q,tc= 'd')
sol = cvxopt.solvers.qp(P,q)
# print('u* = ', sol['x'])
# print('J* = ', sol['primal objective'] + x0.T @ Sx.T @ Qbar @ Sx @ x0)


# %% unconstrained linear finite time optimal control 

nx = np.size(A,0)
nu = np.size(B,1)

P = np.zeros((nx,nx,N+1))
F = np.zeros((nu,nx,N))
# ----------------pay attention to size here----------------

for i in range(N-1,-1,-1):
    # F[:,:,-1] = - inv(B.T @P[:,:,i+1] @B + R)@B.T @P[:,:,i+1] @ A 
    P_k1 = P[:,:,i+1]
    F[:,:,i] = -inv(B.T @ P_k1 @ B +R) @ B.T @ P_k1 @ A
    P[:,:,i] = A.T @ P_k1 @A + Q +A.T @P_k1 @B@ F[:,:,i]

Jopt_DP = x0.T @ P[:,:,0] @ x0
# print('opt cost from recursive approach: ',Jopt_DP)

def sysSim(A,B,D,w,xCurr,uCurr):
    x_next = A @ xCurr + B* uCurr + D * w 
    return x_next

D = np.array([[.1],[.1]])
w = np.random.normal(0,10,N)
# print(N)

x_batch = [x0.reshape((2,1))]
x_recursive = [x0.reshape((2,1))]

for i in range(N):
    x_batch.append(sysSim(A,B,D,w[i],x_batch[0],U0_star[i]))
    x_recursive.append(sysSim(A,B,D,w[i],x_recursive[i],F[:,:,i] @ x_recursive[i]))

t_grid = np.arange(N+1)

x_batch = np.array(x_batch)
x_recursive = np.array(x_recursive)

if False:
    plt.plot(t_grid,x_batch[:,0],x_recursive[:,0])
    plt.legend(['state 1 batch','state 1 dyn'],loc = 'best')
    plt.show()

# %% Constrained finite time optimal control 
Ax = np.array([1,1,0,1]).reshape((2,2))
Bx = np.array([0,1]).reshape((2,1))
Q = np.eye(2)
P = np.eye(2)
R = 0.1 
N = 3 
x0 = np.array([-1,-1]).reshape((2,1))

ULlim = -1
UUlim = 1
xLlim = [-15,-15]
xUlim = [15,15]

Qbar = block_diag(np.kron(np.eye(N),Q),P)
Rbar = np.kron(np.eye(N),R)
Sx, Su = Sx_Su(A,B,N)
QSu = Qbar @ Su
H = Su.T @ QSu +Rbar
F = Sx.T @ QSu
K = -inv(H) @F.T
P0 = F@K + Sx.T @ Qbar @ Sx
nX = 2 

A = np.concatenate([np.kron(np.array([[1],[-1]]),np.eye(3)), Su, -Su], axis = 0)
b = np.concatenate([np.ones((nX*N,1)), 15*np.ones((2*nX*(N+1),1))-np.concatenate([Sx,-Sx], axis = 0)@x0], axis = 0)

c = x0.T @ Sx.T @ Qbar @ Sx @ x0

P = 2*H 
q = 2 * x0.T @ F 

P = cvxopt.matrix(P, tc='d')
q = cvxopt.matrix(q.T, tc='d')
G = cvxopt.matrix(A, tc='d')
h = cvxopt.matrix(b, tc='d')

from ttictoc import tic,toc 

tic()
sol = cvxopt.solvers.qp(P,q,G,h)
t_dense = toc()

Jopt_dense = sol['primal objective'] + c 
print('opt cost= ', Jopt_dense)

