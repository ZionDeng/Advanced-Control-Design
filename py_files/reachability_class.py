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
