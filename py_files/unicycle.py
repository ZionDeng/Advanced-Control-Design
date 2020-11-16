# optimal control of a unicycle with pyomo

import matplotlib.pyplot as plt 
import numpy as np 
 
from pyomo.dae import *
from pyomo.environ import *

Ts = .05
N = 50 
TFinal = Ts*N
Nx = 3 
Nu = 2 

#  x' = vcos(theta); y' = vsin(theta), theta' = w
#  u = [v,w]

import pyomo.environ as pyo 

model = pyo.ConcreteModel()
model.tidx = pyo.Set(initialize = range(N+1))
model.xidx = pyo.Set(initialize = range(Nx))
model.uidx = pyo.Set(initialize = range(Nu))

model.z = pyo.Var(model.xidx, model.tidx)
model.u = pyo.Var(model.uidx, model.tidx)

# the objective can be different 
model.obj = pyo.Objective(
    expr = sum(model.u[0,t] **2 for t in model.tidx if t <N),
    sense = pyo.minimize
)

UdotLim = .03 
z0 = [0,0,0]
zf = [1,1,np.pi/4]

model.cons1 = pyo.Constraint(
    model.xidx, rule = lambda model,i:
    model.z[i,0] == z0[i]
)
model.cons2 = pyo.Constraint(
    model.tidx, rule = lambda model,t:
    model.z[0,t+1] == model.z[0,t] + Ts *(pyo.cos(model.z[2,t])* model.u[0,t]) if t < N else pyo.Constraint.Skip
)
model.cons3 = pyo.Constraint(
    model.tidx, rule =lambda model,t:
    model.z[1,t+1] == model.z[1,t] + Ts* (pyo.sin(model.z[2,t])* model.u[0,t]) if t < N else pyo.Constraint.Skip 
)
model.cons4 = pyo.Constraint(
    model.tidx, rule = lambda model,t: 
    model.z[2,t+1] == model.z[2,t] + Ts * model.u[1,t]
    if t<N else pyo.Constraint.Skip
)

model.cons5 = pyo.Constraint(
    model.tidx, rule = lambda model,t: 
    model.u[0,t] <=1 
    if t<N-1 else pyo.Constraint.Skip
)
# pay attention to N-1 here ----------------
model.cons6 = pyo.Constraint(
    model.tidx, rule = lambda model,t: 
    model.u[0,t] >= -1 
    if t<N-1 else pyo.Constraint.Skip
)
model.cons7 = pyo.Constraint(
    model.tidx, rule = lambda model,t: 
    model.u[1,t] <= 1 
    if t<N-1 else pyo.Constraint.Skip
)
model.cons8 = pyo.Constraint(
    model.tidx, rule = lambda model,t: 
    model.u[1,t] >= -1 
    if t<N-1 else pyo.Constraint.Skip
)
model.cons9 = pyo.Constraint(
    model.xidx, rule = lambda model,i: 
    model.z[i,N] == zf[i]
)
results = pyo.SolverFactory('ipopt').solve(model).write()

z1 = [model.z[0,0]()]
z2 = [model.z[1,0]()]
z3 = [model.z[2,0]()]
u1 = [model.u[0,0]()]
u2 = [model.u[1,0]()]

for t in model.tidx:
    if t<N:
        z1.append(model.z[0,t+1]())
        z2.append(model.z[1,t+1]())
        z3.append(model.z[2,t+1]())
    if t<N-1:
        u1.append(model.u[0,t+1]())
        u2.append(model.u[1,t+1]())

plt.figure(1)
plt.plot(z1,z2,'b')
# plt.show()

m = pyo.ConcreteModel()
m.tf = pyo.Param(initialize = TFinal)
m.t = ContinuousSet(bounds = (0,m.tf))
m.u1 = Var(m.t, initialize = 0)
m.u2 = Var(m.t, initialize = 0)
m.z1 = Var(m.t)
m.z2 = Var(m.t)
m.z3 = Var(m.t)
m.dz1dt = DerivativeVar(m.z1, wrt=m.t)
m.dz2dt = DerivativeVar(m.z2, wrt=m.t)
m.dz3dt = DerivativeVar(m.z3, wrt=m.t)

m.z1dot = Constraint(
    m.t, rule = lambda m,t:
    m.dz1dt[t] == pyo.cos(m.z3[t]) * m.u1[t]
)
m.z2dot = Constraint(
    m.t, rule = lambda m,t:
    m.dz2dt[t] == pyo.sin(m.z3[t]) * m.u1[t]
)
m.z3dot= Constraint(
    m.t, rule = lambda m,t:
    m.dz3dt[t] == m.u2[t]
)
m.cons1 = Constraint(
    m.t, rule = lambda m,t:
    m.u1[t] <= 1 
)
m.cons2 = Constraint(
    m.t, rule = lambda m,t:
    m.u1[t] >= -1 
)
m.cons3 = Constraint(
    m.t, rule = lambda m,t:
    m.u2[t] <= 1 
)
m.cons4 = Constraint(
    m.t, rule = lambda m,t:
    m.u2[t] >= -1 
)
def _init(m):
    yield m.z1[0] == z0[0]
    yield m.z2[0] == z0[1]
    yield m.z3[0] == z0[2]
m.init_conditions = ConstraintList(rule=_init)

def _end(m):
    yield m.z1[m.tf] == zf[0]
    yield m.z2[m.tf] == zf[1]
    yield m.z3[m.tf] == zf[2]
m.end_conditions = ConstraintList(rule=_end)

TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t, nfe=30)
# Solve algebraic model
results = SolverFactory('ipopt').solve(m)


plt.figure(1)
plt.title('trajectory')
plt.plot([value(m.z1[t]) for t in m.t], [value(m.z2[t]) for t in m.t],'o')
plt.show()