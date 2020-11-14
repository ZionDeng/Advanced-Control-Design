# nonlinear constrained finite time optimal control 

import matplotlib.pyplot as plt 
import numpy as np 
import pyomo.environ as pyo 

Tau = 0.2 
Gamma = 10 
Ts = 0.05 
N = 120 
TFianl = Ts * N 

Nx = 2
Nu = 1 
UdotLim = 0.03

def disc_dyn(x,u):
    x_next = np.empty((Nx,))
    x_next[0] = x[0] + Ts*(np.sin(x[0]) + Gamma * np.arctan(x[1]))
    x_next[1] = x[1] + Ts / Tau * (x[1] - u)
    return x_next

tValues = [0,3,3.5,5.5,6]
xDesValues = [0, 0.75*np.pi/4, 0.67*np.pi/4, 1.25*np.pi/4, 1.25*np.pi/4]

from scipy import interpolate

f = interpolate.interp1d(tValues,xDesValues,'linear')
tGrid = np.linspace(tValues[0],tValues[-1],N+1)
xDesired = f(tGrid)

# plt.scatter(tGrid,xDesired)
# plt.show()

model = pyo.ConcreteModel()
model.tidx = pyo.Set(initialize = range(N+1))
model.xidx = pyo.Set(initialize = range(Nx))
model.uidx = pyo.Set(initialize = range(Nu))

model.x = pyo.Var(model.xidx, model.tidx) 
model.u = pyo.Var(model.uidx, model.tidx)

model.cost = pyo.Objective(
    expr = sum((model.x[0,t] - xDesired[t])**2 for t in model.tidx if t<N),
    sense = pyo.minimize
)

# constraints 
model.cons1 =pyo.Constraint(
    model.xidx, rule = lambda model, i:
    model.x[i,0] == 0
)
model.cons2 = pyo.Constraint(
    model.tidx, rule = lambda model,t:
    model.x[0,t+1] == model.x[0,t]+ Ts * (pyo.sin(model.x[0,t]) + Gamma * pyo.atan(model.x[1,t])) if t<N else pyo.Constraint.Skip
)
model.cons3 = pyo.Constraint(
    model.tidx, rule = lambda model,t:
    model.x[1,t+1] == model.x[1,t] + Ts/Tau * model.x[1,t] - model.u[0,t]
    if t < N else pyo.Constraint.Skip
)

model.cons4 = pyo.Constraint(
    model.tidx, rule = lambda model,t:
    model.u[0,t+1] - model.u[0,t] >= -Ts * UdotLim
    if t<N-1 else pyo.Constraint.Skip
)

model.cons5 = pyo.Constraint(
    model.tidx, rule = lambda model,t:
    model.u[0,t+1] - model.u[0,t] <= Ts * UdotLim
    if t<N-1 else pyo.Constraint.Skip
)

# model.cons6 = pyo.Constraint(
#     expr = model.x[0,N] -xDesired[N] <= 0.025* xDesired[N]
# )

# model.cons7 = pyo.Constraint(
#     expr =  -0.025* xDesired[N] <= model.x[0,N] -xDesired[N] 
# )
model.constraint6 = pyo.Constraint(expr = 0.975*xDesired[N] - model.x[0, N] <= 0.0)
model.constraint7 = pyo.Constraint(expr = model.x[0, N] - 1.025*xDesired[N] <= 0.0)

result = pyo.SolverFactory('ipopt').solve(model)

x1 = [model.x[0,0]()]
x2 = [model.x[1,0]()]
u = [model.u[0,0]()]

for t in model.tidx:
    if t<N:
        x1.append(model.x[0,t+1]())
        x2.append(model.x[1,t+1]())
    if t<N-1:
        u.append(model.u[0,t+1]())

# plt.figure()
# plt.plot(tGrid, x1,'b')
# plt.plot(tGrid, x2,'g')
# plt.plot(tGrid[0:-1], u,'r')
# plt.show()


x_actual = np.zeros((Nx,N+1))
for t in range(N):
    x_actual[:,t+1] = disc_dyn(x_actual[:,t],u[t])

plt.figure()
plt.plot(tGrid,x_actual[0,:],'b')
plt.plot(tGrid,xDesired,'g')
plt.plot(tGrid,x1,'--r')
plt.legend(['Actual','Desired','open-loop'])
plt.xlabel('Time')
plt.ylabel('x1 Trajectory')
plt.show()

# --------------result is abnormal -------------