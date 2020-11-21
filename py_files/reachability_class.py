import polytope as pt
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

A = np.array([[1, 0],
              [0, 1],
              [0, -1],
              [-1, 0]])

b = np.array([[10],
              [10],
              [10],
              [10]])

P = pt.Polytope(A,b)
if False:
    fig, ax = plt.subplots(1,1)
    plt.rcParams['figure.figsize'] = [20, 20]
    P.plot(ax, color='r')
    ax.autoscale_view()
    ax.axis('equal')
    plt.show()

# reduce 
P = pt.reduce(P)
print(P)

# HV conversion 
V=np.array([[10,10],[-10,10],[10,-10],[-10,-10]])
P = pt.qhull(V)
print(P)

V1 = pt.extreme(P)
print(V1)


# Minkwoski sum of two Polytopes
def minkowski_sum(X,Y):
    v_sum = []
    if isinstance(X,pt.Polytope):
        X = pt.extreme(X) # make sure it is V version

    if isinstance(Y,pt.Polytope):
        Y = pt.extreme(Y)
    
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            v_sum.append(X[i,:]+Y[j,:])
    return pt.qhull(np.asarray(v_sum))


P = np.array([[1, 0],
              [0, 1],
              [0, -1],
              [-1, 0]])

p = np.array([[6],
              [6],
              [6],
              [6]])

Q = np.array([[1, 0],
              [0, 1],
              [0, -1],
              [-1, 0]])

q = np.array([[2],
              [2],
              [2],
              [2]])

Pp = pt.Polytope(P, p)
Qq = pt.Polytope(Q, q)

p_sum  = minkowski_sum(Pp, Qq)

if False:
    fig, ax = plt.subplots(1,1)
    p_sum.plot(ax, color='b')
    Pp.plot(ax, color='r')
    Qq.plot(ax, color='g')
    ax.legend(['sum', 'P', 'Q'])
    ax.autoscale_view() 
    ax.axis('equal')
    plt.show()

def projection(X,nx):
    V_sum = []
    V = pt.extreme(X)
    for i in range(V.shape[0]):
          V_sum.append(V[i,0:nx])
    return pt.qhull(np.asarray(V_sum))

P = np.array([[1, 0, 0],
              [0, 1, 0 ],
              [0, -1, 0 ],
              [-1, 0, 0],
              [0,0,1],
              [0,0,-1]])

p = np.array([[6],
              [3],
              [3],
              [6],
              [10],
              [10]])

Pp = pt.Polytope(P, p)

PProj  = projection(Pp,2)
# print(PProj)

# %% N steps controllable sets to a given set 

def precursor(Sset,A,Uset = pt.Polytope(),B = np.array([])):
    # see definition of Pre(S) in slides
    if not B.any(): # if B is nothing
        return pt.Polytope(Sset.A @ A ,Sset.b)
    else:
        tmp = minkowski_sum(Sset,pt.extreme(Uset) @ -B.T)
    return pt.Polytope(tmp.A @ A, tmp.b)

# Example one step 
A = np.array([[1.5, 0],
              [1.0, -1.5]])

B = np.array([[1.0], 
              [0.0]])

S = pt.Polytope(np.array([[1.0, 0], 
                          [0, 1.0],
                          [-1, 0],
                          [0, -1]]), 
                np.array([[1], 
                          [1],
                          [1],
                          [1]]))

U = pt.Polytope(np.array([[1.0], 
                          [-1.0]]),
                np.array([[5.0], 
                          [5.0]]))
if False:
    preS = precursor(S, A, U, B)
    fig, ax = plt.subplots()
    S.plot(ax, color='b')
    #preS.intersect(S).plot(ax, color='r')
    preS.plot(ax, color='r')
    ax.legend(['S', 'Pre(S)'])
    plt.rcParams['figure.figsize'] = [10, 10]
    ax.autoscale_view()
    ax.axis('equal')
    plt.show()

# example in 10 steps 
# Example 10 steps 


N = 10  # number of steps
K = {}
PreS = precursor(S, A, U, B) #one step controllable to S
for j in range(N):
    K[j]= PreS #for j=0 one ste controllable
    PreS = precursor(K[j], A, U, B)


# Plotting 
plt.clf()
plt.cla()
plt.close('all')
if False:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    S.plot(ax, color='b')
    K[0].plot(ax, color='g', alpha=0.1, linestyle='solid', linewidth=1, edgecolor=None)      # K_0 is equivalent to Pre S
    K[1].plot(ax, color='r', alpha=0.1, linestyle='solid', linewidth=1)                      # K_1 two step controllable set 
    K[2].plot(ax, color='r', alpha=0.1, linestyle='solid', linewidth=1)                      # K_2 three step controllable set 
    K[3].plot(ax, color='r', alpha=0.1, linestyle='solid', linewidth=1)                      # K_3 
    K[4].plot(ax, color='r', alpha=0.1, linestyle='solid', linewidth=1)                      # K_4 
    K[N-1].plot(ax, color='b', alpha=0.1, linestyle='solid', linewidth=1)                      # K_5  
    ax.legend(['K0', 'K1', 'K2', 'K3', 'K4', 'KN-1'])

    plt.rcParams['figure.figsize'] = [10, 10]
    ax.autoscale_view()
    ax.axis('equal')
    plt.show()


K=np.array([[-0.1,-0.1]])
Acl=A+B@K # x+=Ax+Bu but u=K*x-? x+=(A+BK)x
eig_val = np.linalg.eigvals(Acl)
print('Eigen Values are',eig_val,', if they are <0: ', np.all(eig_val<0))

S = pt.Polytope(np.array([[1.0, 0], 
                          [0, 1.0],
                          [-1, 0],
                          [0, -1]]), 
                np.array([[1], 
                          [1],
                          [1],
                          [1]]))

# Input Constraints Hu*u<=Ku (umin=-5, umax=5)
Hu=np.array([[1.0],[-1.0]])
ku=np.array([[5.0], [5.0]])
# recall U = pt.Polytope(Hu,ku)

# if u has to be in U then Kx has to be in U -> (Hu*K)*x<=ku
# U now become X constraints
X = pt.Polytope(Hu@K,ku)
Snew=S.intersect(X)
preS = precursor(Snew, Acl)

if False:
    fig, ax = plt.subplots()
    Snew.plot(ax, color='b')
    # X.plot(ax, color='g')  unbounded X 
    #preS.intersect(S).plot(ax, color='r')
    preS.plot(ax, color='r')
    ax.legend(['S', 'Pre(S)','X'])
    plt.rcParams['figure.figsize'] = [5, 5]
    ax.autoscale_view()
    ax.axis('equal')
    plt.title('Closed S')
    plt.show()

def max_pos_inv(A, S):
    maxIterations = 500
    # initialization
    Omega_i = S 
    for i in range(maxIterations):
        # compute backward reachable set
        P = precursor(Omega_i, A)
        # intersect with the state constraints
        P = pt.reduce(P).intersect(Omega_i)
        if P == Omega_i:
            Oinf = Omega_i
            break
        else:
            Omega_i = P
    if i == maxIterations:
        converged = 0
    else:
        converged = 1
    return Oinf, converged

def max_cntr_inv(A,B,X,U):
    maxIterations = 500
    # initialization
    Omega0 = X 
    for i in range(maxIterations):
        # compute backward reachable set
        P = precursor(Omega0, A, U, B)
        # intersect with the state constraints
        P = pt.reduce(P).intersect(Omega0)
        if P == Omega0:
            Cinf = Omega0
            break
        else:
            Omega0 = P
    if i == maxIterations:
        converged = 0
    else:
        converged = 1
    return Cinf, converged


A = np.array([[0.5, 0],
              [1.0, -0.5]])

X = pt.Polytope(np.array([[1.0, 0], 
                          [0, 1.0],
                          [-1, 0],
                          [0, -1]]), 
                np.array([[10.0], 
                          [10.0],
                          [10.0],
                          [10.0]]))

Oinf, converged = max_pos_inv(A,S)

if False:
    fig, ax = plt.subplots()
    Oinf.plot(ax, color='g', alpha=0.5, linestyle='solid', linewidth=1, edgecolor=None)
    ax.autoscale_view()
    ax.axis('equal')
    plt.show()

# Example 10.6 (Figure 10.8) MPC book

A = np.array([[1.5, 0],
              [1.0, -1.5]])

B = np.array([[1.0], 
              [0.0]])
X = pt.Polytope(np.array([[1.0, 0], 
                          [0, 1.0],
                          [-1, 0],
                          [0, -1]]), 
                np.array([[10.0], 
                          [10.0],
                          [10.0],
                          [10.0]]))

U = pt.Polytope(np.array([[1.0], 
                          [-1.0]]),
                np.array([[5.0], 
                          [5.0]]))

Cinfset, converged = max_cntr_inv(A, B, X, U)
if False:
    fig, ax = plt.subplots()
    X.plot(ax, color='b')
    Cinfset.plot(ax, color='r')
    ax.legend(['X', 'C_inf'])
    ax.autoscale_view()
    ax.axis('equal')
    plt.show()    