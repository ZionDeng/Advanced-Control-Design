import numpy as np 
import scipy as cp 
from scipy.optimize import linprog

import cvxopt

f = np.ones((3,))   # 1D array
A = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1], [-1, 1, -1]])   # 2D array
b = np.array([-2,1,3,-4])   # 1D array
# default bound is (0, None), so here we need to specify bound as (None, None), 
# since all the bounds all already considered in the inequality constraints 
opt1 = cp.optimize.linprog(c=f, A_ub=A, b_ub=b, bounds=(None, None), method='simplex')
# print('x*=', opt1.x)
# print('J*=', opt1.fun)

# %% qp problems 
n = 4 # dimenstion of x
A = np.array([[-0.54, -1.81, 0.25, -0.46],
 [-0.38, 0.37, -2.48, -0.68],
 [-1.31, 0.74, -1.57, 0.28],
 [-0.31, -0.02, 0.75, 0.20]])
b = np.array([ 0.40, -2.45, -0.23, 0.98])
l_i = -0.5 
u_i = 0.5

P = 2 * A.T @A 
q = -2 * A.T @b
G = np.concatenate([np.eye(4), -np.eye(4)],axis= 0)
h = np.concatenate([np.ones((n,1))*u_i,np.ones((n,1))*-l_i],axis=0)

P = cvxopt.matrix(P,tc = 'd')
q = cvxopt.matrix(q,tc = 'd')
G = cvxopt.matrix(G,tc = 'd')
h = cvxopt.matrix(h,tc = 'd')
sol = cvxopt.solvers.qp(P,q,G,h)
# print('x*=', sol['x'])
# print('p*=', sol['primal objective'] + b.T @b)
# ----------pay attention to b.T @b here -----------

xstar = (np.linalg.inv((A.T @ A))) @ (A.T.dot(b))    # Analytical solution of unconstrained least-squares 
xstar[xstar > u_i] = u_i  # projection: set any entries that are greater than 0.5 to 0.5
xstar[xstar < l_i] = l_i  # projection: set any entries that are less than -0.5 to -0.5

# Compare performance (i.e. cost function)
# print('x_analytical:', xstar)  # analytical solution with projection 
# print('x_cvxopt:', sol['x']) # direct solution from cvxopt
# print(np.linalg.norm(A @ np.reshape(sol['x'],(4,)) - b))
# print(np.linalg.norm(A @ xstar - b))

# %% Problem2 
from scipy.io import loadmat 

Data = loadmat('Data/etch2169.mat')
RoomTempData = Data['RoomTempData']
FanData = Data['FanData']
SupplyTempData = Data['SupplyTempData']


from datetime import date
from datetime import datetime as dt

# datenum is a function which converts date stings and date vectors into serial date numbers.
# Date numbers are serial days elapsed from some reference time. 

def datenum(d):
    return 366 + d.toordinal() + (d - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)

d_start = dt.strptime('2018-9-9 10:1','%Y-%m-%d %H:%M')
d_end = dt.strptime('2018-9-9 15:59','%Y-%m-%d %H:%M')
d_start_plus_onemin = dt.strptime('2018-9-9 10:2','%Y-%m-%d %H:%M')

TS = datenum(d_start_plus_onemin) - datenum(d_start)
TimeQuery = np.arange(start=datenum(d_start), stop=datenum(d_end), step=TS)

import matplotlib.pyplot as plt 
if False:
    counts, bins = np.histogram(np.diff(RoomTempData[:,0]))
    plt.hist(bins[:-1],bins,weights=counts)
    # plt.show()

from scipy import interpolate

xData = np.cumsum(np.hstack([np.zeros((1,)), 0.1+np.random.rand(19,)]))
yTmp = np.hstack([np.sort(3*np.random.rand(10,1)), np.fliplr(np.sort(3*np.random.rand(10,1)))])
yData = yTmp.flatten()
xQuery = np.arange(start=0.1, stop=np.max(xData), step=0.2)

f_interp = interpolate.interp1d(xData, yData, 'linear')
yInterpLinear = f_interp(xQuery)

f_spline = interpolate.UnivariateSpline(xData, yData)
yInterpSpline = f_spline(xQuery)

if False:
    plt.subplot(2,1,1)
    plt.plot(xData, yData, 'k*', xQuery, yInterpLinear, '-or')
    plt.legend(['Data Points', 'Linear Interpolated Values'])
    plt.ylabel('y')
    plt.subplot(2,1,2)
    plt.plot(xData, yData, 'k*', xQuery, yInterpSpline, '-or')
    plt.legend(['Data Points', 'Spline Interpolated Values'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

time_T = RoomTempData[:,0]
time_u1 = FanData[:,0]
time_u2 = SupplyTempData[:,0]

data_T =RoomTempData[:,1]
data_u1 = FanData[:,1]
data_u2 = SupplyTempData[:,1]

# spline interpolate 
fun_T= interpolate.UnivariateSpline(time_T,data_T)
spline_T = fun_T(TimeQuery)

fun_u1 = interpolate.UnivariateSpline(time_u1,data_u1)
spline_u1 = fun_u1(TimeQuery)

fun_u2 = interpolate.UnivariateSpline(time_u2,data_u2)
spline_u2 = fun_u2(TimeQuery)

if False:
    plt.plot(TimeQuery,spline_T,'-or')
    plt.plot(time_T,data_T)
    plt.show()

N = 360 
def bldgIdentification(Tdata,u1Seq,u2Seq):
    b = -np.diff(Tdata)
    # a2 = - u1Seq @ u2Seq
    # a1 = u1Seq @ Tdata[0:-1]
    a2 = -np.multiply(u1Seq,u2Seq)
    a1 = np.multiply(u1Seq,Tdata[0:-1])
# --------can only use multiply here ------------
    A = np.concatenate([np.reshape(a1,(len(a1),1)),np.reshape(a2,(len(a2),1))],axis= 1)
    P = 2 * A.T @ A 
    q = -2 * A.T @b
    P = cvxopt.matrix(P,tc='d')
    q = cvxopt.matrix(q,tc='d')
    sol = cvxopt.solvers.qp(P,q)
    estParm = sol['x']
    return estParm

Tdata = RoomTempData[:,1]
u1Seq = FanData[0:-1,1]
u2Seq = SupplyTempData[0:-1,1]
# print(bldgIdentification(Tdata,u1Seq,u2Seq))

# %% P4

def buildLinClass(G1,G2):
    nf, p1 = np.shape(G1)
    _, p2 = np.shape(G2)
    P = p1+p2
    
    # c matrix for cost function: c* (c1,c2,b,v1,,,vP).T 
    c = np.concatenate([np.zeros(nf+1),np.ones(P)],axis= 0)

    # A matrix for 3 constraints : tk >=0, c.T * vk + tk >=1; c.T * vk + b -tk <= -1

    A = np.concatenate(
        [
            np.concatenate([np.zeros((P,nf+1)),-np.eye(P)],axis= 1),
            np.concatenate([-G1.T,-np.ones((p1,1)),-np.eye(p1),np.zeros((p1,p2))],axis=1),
            np.concatenate([G2.T,np.ones((p2,1)),np.eye(p2),np.zeros((p2,p1))],axis=1)
        ],axis= 0
    )
    b = np.concatenate(
        [np.zeros((P,1)),-np.ones((p1,1)),-np.ones((p2,1))],
        axis= 0
    )

    c = cvxopt.matrix(c,tc = 'd')
    A = cvxopt.matrix(A,tc = 'd')
    b = cvxopt.matrix(b,tc = 'd')

    sol = cvxopt.solvers.lp(c,A,b)
    xOpt = sol['x']
    c = np.array(xOpt[:nf]).flatten()
    b = np.array(xOpt[nf]).flatten()
    t = np.array(xOpt[nf+1:]).flatten()

    return c,b,t

nF = 2
nP = 100
cTrue = np.random.randn(nF,1)
bTrue = np.random.randn(1,1)
Pop = np.random.randn(nF, nP) 

LPop =  cTrue.T@Pop + bTrue
idx_pos = np.argwhere(LPop>0)
idx_neg = np.argwhere(LPop<0)

G1 = Pop[:, idx_pos[:,1]]  # create the populations based on their L-value
G2 = Pop[:, idx_neg[:,1]]  # create the populations based on their L-value

[cEst, bEst, tAdjust] =  buildLinClass(G1,G2)

max(abs(tAdjust))      # should be 0 (or very close)
f1Min = np.min(Pop[0,:])  # minimum age
f1Max = np.max(Pop[0,:])  # maximum age
f2Min = np.min(Pop[1,:])  # minimum number of movies
f2Max = np.max(Pop[1,:])  # maximum number of movies

if False:
    plt.plot(np.array([f1Min-1, f1Max+1]), -(cEst[0]*np.array([f1Min-1, f1Max+1])+bEst)/cEst[1],'b')
    plt.plot(G1[0,:],G1[1,:],'b*')
    plt.plot(G2[0,:],G2[1,:],'ro')
    plt.xlim([f1Min-0.1, f1Max+0.1])
    plt.ylim([f2Min-0.1, f2Max+0.1])
    plt.axis('equal')   
    plt.show()

if False:
    nF = 2
    nP = 100
    nOut = np.int(0.1*nP)
    cTrue = np.random.randn(nF,1)
    bTrue = np.random.randn(1,1)
    Pop = np.random.randn(nF, nP)
    Noise = np.asarray([np.random.randn(1) if i < nOut else 0.0 for i in range(nP)], dtype=object)

    LPop =  cTrue.T@Pop + bTrue + Noise  #corrupt some L-values with noise
    idx_pos = np.argwhere(LPop>0)
    idx_neg = np.argwhere(LPop<0)

    G1 = Pop[:, idx_pos[:,1]]  # create the populations based on their L-value
    G2 = Pop[:, idx_neg[:,1]]  # create the populations based on their L-value

    [cEst, bEst, tAdjust] =  buildLinClass(G1,G2)

    max(abs(tAdjust))      # likely nonzero, and > 1, dealing wiht non-separability  
    f1Min = min(Pop[0,:])  # minimum age
    f1Max = max(Pop[0,:])  # maximum age
    f2Min = min(Pop[1,:])  # minimum number of movies
    f2Max = max(Pop[1,:])  # maximum number of movies

    plt.plot(np.array([f1Min-1, f1Max+1]), -(cEst[0]*np.array([f1Min-1, f1Max+1])+bEst)/cEst[1],'b')
    plt.plot(G1[0,:],G1[1,:],'b*')
    plt.plot(G2[0,:],G2[1,:],'ro')
    plt.xlim([f1Min-0.1, f1Max+0.1])
    plt.ylim([f2Min-0.1, f2Max+0.1])
    plt.axis('equal')
    plt.show()

# P8 
# test = np.arange(12).reshape((2,6))
# print(np.size(test,1))

def reg1Inf(A1,b1,Ainf,binf,Ac,bc):
    # the question transformed to:
    # min(z1,z2,,zn,t1,t2,,t_nc,t_inf){t1,t2,,t_nc,t_inf}
    # t1,t2,,t_nc = |A1 *z -b1|
    # t_inf = max(|A_inf *z - b_inf|)

    n_1, nx = np.shape(A1)
    n_inf, _ = np.shape(Ainf)
    nc, _ = np.shape(Ac)

    c = np.concatenate(
        [
            # np.zeros(np.size(Ac,1)),np.ones(np.size(A1,0)),np.array((1,))
            np.zeros(nx),np.ones(n_1),np.array((1,))
        ],axis= 0
    )
    A = np.concatenate(
        [
            # np.concatenate([A1,-np.eye(np.size(A1,0)),np.zeros((np.size(A1,0),1))],axis=1),
            # np.concatenate([-A1,-np.eye(np.size(A1,0)),np.zeros((np.size(A1,0),1))],axis=1),
            # np.concatenate([Ainf,np.zeros((np.size(Ainf,1),np.size(A1,1))),np.ones((np.size(Ainf,0),1))],axis= 1),
            # np.concatenate([-Ainf,np.zeros((np.size(Ainf,1),np.size(A1,1))),-np.ones((np.size(Ainf,0),1))],axis= 1),
            # np.concatenate([Ac,np.zeros((np.size(Ac,0),np.size(A1,0))),np.zeros((np.size(Ac,0),1))],axis=1)
            np.concatenate([A1,-np.eye(n_1),np.zeros((n_1,1))],axis=1),
            np.concatenate([-A1,-np.eye(n_1),np.zeros((n_1,1))],axis=1),
            np.concatenate([Ainf,np.zeros((n_inf,nx)),np.ones((n_inf,1))],axis= 1),
            np.concatenate([-Ainf,np.zeros((n_inf,nx)),-np.ones((n_inf,1))],axis= 1),
            np.concatenate([Ac,np.zeros((nc,n_1)),np.zeros((nc,1))],axis=1)

        ],axis = 0
    )
    b = np.concatenate([b1,-b1,binf,-binf,bc],axis=0)

    c = cvxopt.matrix(c,tc='d')
    A = cvxopt.matrix(A,tc='d')
    b = cvxopt.matrix(b,tc='d')


    sol = cvxopt.solvers.lp(c,A,b)
    xOpt = sol['x']
    J = sol['primal objective']
    return xOpt,J 

a1 = np.zeros((3,3))
b1 = np.array([0,0,0]).reshape((3,1))
ainf = np.array([[2,0,-1],[1,-1,0]])
binf = np.array([1,2]).reshape((2,1))
ac = np.array([[1,-1,1],[-1,-1,0]])
bc = np.array([-1,-1]).reshape((2,1))
xOpt, J = reg1Inf(a1,b1,ainf,binf,ac,bc)
print('xOpt: ',xOpt)
print('J* = ',J)