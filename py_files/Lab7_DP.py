# constrained finite time optimal control 

import numpy as np 
from scipy import interpolate
import multiprocessing

# pool = multiprocessing.Pool(4)
# print(np.array(range(3)).reshape((3,1)))

A = np.array([1,1,0,1]).reshape((2,2))
B = np.array([0,1]).reshape((2,1))

NX = 2 
NU = 1 


def f_dyn(x,u):
    return A @ np.array([x]).reshape((NX,1)) + B @ np.array([u]).reshape((NU,1))

xmin = -15
xmax = 15 

umin = -1 
umax = 1 

N = 3 # horizon 

Q = np.eye(NX)
R = np.array([0.1]).reshape((NU,NU))
PN = Q 

# create function for state cost and 'optimal cost-to-go'
# cost-to-go is a  function of the state and changes with time, 
# store this as a dictionary 

def J_stage(x,u):
    x = np.array([x]).reshape(NX,1)
    u = np.array([u]).reshape(NU,1)
    return  x.T @ Q @ x + u.T @ R @ u # return state cost 

Jopt = {}
Uopt = {}

# grid the x under the constraint 
Nx_grid = 50 
x1_grid = np.linspace(xmin,xmax,Nx_grid)
x2_grid = np.linspace(xmin,xmax,Nx_grid)
points = (x1_grid,x2_grid)

# grid the input space 
Nu_grid = 21 
u_grid = np.linspace(umin,umax,Nu_grid)

# Allocate memory for Jopt and Uopt arrays
# Jopt[N] is a known, quadratic function (from PN).
Jopt_array = np.nan * np.zeros((Nx_grid,Nx_grid,N+1))
Uopt_array = np.nan * np.zeros((Nx_grid,Nx_grid,N))

for idx1,x1 in enumerate(x1_grid):
    for idx2,x2 in enumerate(x2_grid):
        x = np.array([x1,x2]).reshape(NX,)
        Jopt_array[idx1,idx2,N] = x.T @ PN @ x

# interpolate the Jopt and Uopt
def Jopt_interpolate(x,j):
    return interpolate.interpn(points,Jopt_array[:,:,j],x.flatten())
def Uopt_interpolate(u,j):
    return interpolate.interpn(points,Uopt_array[:,:,j],x.faltten())




for j in reversed(range(N)):
    print('Computing J:',j)

    # initialize Jopt_array-1 and uPolicy[j] matrix at iteration
    # loop over the 2-D grid in X

    for idx1, x1 in enumerate(x1_grid):
        for idx2,x2 in enumerate(x2_grid):
            xi = np.array([x1,x2]).reshape(NX,)

            def fun(u):
                J_j = J_stage(xi,u).flatten() + Jopt_interpolate(f_dyn(xi,u),j+1).flatten()
                return J_j

            J_best = np.inf
            U_best = np.NaN
            for u_val in u_grid:
                xi_next = f_dyn(xi,u_val)
                if np.all(xi_next >= xmin) and np.all(xi_next <= xmax):
                    J_act = fun(u_val)
                    if J_act < J_best:
                        U_best = u_val
                        J_best = J_act
                        Jopt_array[idx1,idx2,j] = J_best
                        Uopt_array[idx1,idx2,j] = U_best
x = np.array([-1,-1])
print(Jopt_interpolate(x,0))


from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

fig, ax = plt.subplots(1, N, figsize=(12, 6))
for j in range(N):
    ax[j].contour(Jopt_array[:,:,j], 20, cmap=cm.RdBu, extent=[-15, 15, -15, 15])
    ax[j].set_xlim(-15, 15)
    ax[j].set_ylim(-15, 15)
    ax[j].axis('square')
    plt.xlabel('x1')
    plt.ylabel('x2')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

fig = plt.figure(figsize=(12, 6))
for j in range(N):
    ax = fig.add_subplot(1, N, j+1)
    ax.imshow(Jopt_array[:,:,j], origin='lower', extent=(-15,15,-15,15))
    ax.axis([-15, 15, -15, 15])
    plt.xlabel('x1')
    plt.ylabel('x2')
    
fig = plt.figure(figsize=(12, 6))
for j in range(N):
    ax = fig.add_subplot(1, N, j+1)
    ax.imshow(Uopt_array[:,:,j], origin='lower', extent=(-15,15,-15,15))
    ax.axis([-15, 15, -15, 15])
    plt.xlabel('x1')
    plt.ylabel('x2')

plt.show()
