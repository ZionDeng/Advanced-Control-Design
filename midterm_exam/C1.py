
import matplotlib.pyplot as plt
import numpy as np 
import pyomo.environ as pyo

N = 50
xNbar = 1
x0 =0 
m = pyo.ConcreteModel()
m.tidx = pyo.Set(initialize = range(N+1))
m.u = pyo.Var(m.tidx)
m.x = pyo.Var(m.tidx)

m.cost = pyo.Objective(
    expr = sum((m.x[t]-xNbar) ** 2 for t in m.tidx),
    sense= pyo.minimize 
    
)

m.c1 = pyo.Constraint(
    m.tidx, rule = lambda m,t:
    m.x[t+1] == pyo.sin(m.x[t]) + m.u[t]
    if t < N else pyo.Constraint.Skip
)
m.c21 = pyo.Constraint(
    m.tidx, rule = lambda m,t:
    m.u[t] <= -0.2
    if t<N else pyo.Constraint.Skip
)
m.c22 = pyo.Constraint(
    m.tidx, rule = lambda m,t:
    m.u[t] >= 0.2
    if t<N else pyo.Constraint.Skip
)
m.c31 = pyo.Constraint(
    expr = m.x[N] -xNbar <= -0.1
)
m.c32 = pyo.Constraint(
    expr = m.x[N] -xNbar >= 0.1
)
m.c4 = pyo.Constraint(expr = m.x[0] == x0)

results = pyo.SolverFactory('ipopt').solve(m).write()

x= [m.x[0]()]
u= [m.u[0]()]
for t in m.tidx:
    if t< N: 
        x.append(m.x[t]())
        
    if t< N-1:
        u.append(m.u[t]())
plt.plot(x)
plt.show()
