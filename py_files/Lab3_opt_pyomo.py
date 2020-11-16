# use Pyomo to solve optimization problems 
# linear, quadratic, nonlinear, mixed-integer 

import pyomo.environ as pyo 

# %% linear programming 
model = pyo.ConcreteModel()
model.x = pyo.Var()
model.y = pyo.Var()
model.z = pyo.Var()

model.Obj = pyo.Objective(expr = model.x + model.y + model.z)
model.cons1 = pyo.Constraint(expr = -2<= model.x)
model.cons2 = pyo.Constraint(expr = -1<= model.y)
model.cons3 = pyo.Constraint(expr = -3<= model.z)
model.cons4 = pyo.Constraint(expr = model.x - model.y + model.z >= 4)

solver = pyo.SolverFactory('cbc')
result = solver.solve(model)

# print('x*_solver = ',pyo.value(model.x))
# print('y*_solver = ',pyo.value(model.y))
# print('z*_solver = ',pyo.value(model.z))
# print('opt_value = ',pyo.value(model.Obj))


# %% nonlinear programming 
import numpy as np 

model = pyo.ConcreteModel()
model.z1 = pyo.Var()
model.z2 = pyo.Var()
model.Obj = pyo.Objective(expr = 3 * pyo.sin(-2*np.pi * model.z1) + 2* model.z1 + 4 + pyo.cos(2*np.pi*model.z2) + model.z2)

model.cons1 = pyo.Constraint(expr = (-1,model.z1,1))
model.cons2 = pyo.Constraint(expr = (-1,model.z2,1))
# model.cons2 = pyo.Constraint(expr = -1<= model.z2 <= 1)


results = pyo.SolverFactory('ipopt').solve(model)

# print('zOpt = ',[pyo.value(model.z1),pyo.value(model.z2)])
# print('JOpt = ', pyo.value(model.Obj))

z1 = []
z2 = []
J = []
for _ in range(10):
    z1_init = np.random.uniform(low = -1.0,high = 1.0)
    z2_init = np.random.uniform(low=-1.0, high = 1.0)
    model = pyo.ConcreteModel()
    model.z1 = pyo.Var(initialize = z1_init)
    model.z2 = pyo.Var(initialize = z2_init)
    model.obj = pyo.Objective(expr =3*pyo.sin(-2*np.pi*model.z1) + 2*model.z1 + 4 + pyo.cos(2*np.pi*model.z2) + model.z2)
    model.cons1 = pyo.Constraint(expr = (-1,model.z1,1))
    model.cons2 = pyo.Constraint(expr = (-1,model.z2,1))

    results = pyo.SolverFactory('ipopt').solve(model)
    z1.append(pyo.value(model.z1))
    z2.append(pyo.value(model.z2))
    J.append(pyo.value(model.obj))

# print('z1Opt = ',z1)

import matplotlib.pyplot as plt 

z1_opt = z1 
z2_opt = z2 
fig, ax = plt.subplots(figsize = (15,15))
z = np.linspace(-1,1,100)
z1,z2 = np.meshgrid(z,z)
C = 3* np.sin(-2* np.pi * z1 ) + 2*z1 + 4+ np.cos(2* np.pi* z2) + z2
contour = ax.contour(z1,z2,C, cmap = plt.cm.RdBu, vmin = abs(C).min(), vmax = abs(C).max(),)
ax.clabel(contour,fontsize = 10, inline =1)
ax.axis('square')
ax.scatter(z1_opt,z2_opt,c='r',marker ='o')
# plt.show()

# %% mixed integer programming 

# power plant problem 

# Horizon = 48 
# T = np.array([t for t in range(Horizon)])
T = 48

# predicted demand
d = np.array([100 + 50 * np.sin(t * 2*np.pi/24) for t in range(T)])

N = 3
# N = np.array([n for n in range(Nplant)])

Pmax = [100,50,25]
Pmin = [20,40,1]
C = [10,20,20] 

model = pyo.ConcreteModel()
model.N = pyo.Set(initialize = range(N))
model.T = pyo.Set(initialize = range(T)) 

# production
model.x = pyo.Var(model.N, model.T)

# on/off 
model.u = pyo.Var(model.N, model.T, domain= pyo.Binary)

# cost function 
model.cost = pyo.Objective(
    expr = sum(model.x[n,t]* C[n] for t in model.T for n in model.N),
    sense = pyo.minimize 
)

# demand constraints 
model.demand = pyo.Constraint(
    model.T, rule = lambda model,t: sum(model.x[n,t] for n in range(N)) >= d[t]
)

# production constraints still confusing ???
model.lb = pyo.Constraint(
    model.N, model.T, rule = lambda
    model, n, t: Pmin[n] * model.u[n,t] <= model.x[n,t]
)


model.ub = pyo.Constraint(
    model.N, model.T, rule = lambda
    model,n,t: Pmax[n] * model.u[n,t] >= model.x[n,t]
)

result = pyo.SolverFactory('glpk').solve(model) 
# ---------pay attention to the solver ------------

unit1 = [pyo.value(model.x[0,0])]
unit2 = [pyo.value(model.x[1,0])]
unit3 = [pyo.value(model.x[2,0])]

for t in range(T):
    unit1.append(pyo.value(model.x[0,t]))
    unit2.append(pyo.value(model.x[1,t]))
    unit3.append(pyo.value(model.x[2,t]))
plt.figure(figsize = (15,15))
plt.step(unit1,'b')
plt.step(unit2,'g')
plt.step(unit3,'r')
# plt.show()


# %% Quadratic programming 

x0 = 10 

model = pyo.ConcreteModel()

model.idxx = pyo.Set(initialize = [0,1])
model.idxu = pyo.Set(initialize = [0,1])
model.x = pyo.Var(model.idxx)
model.u = pyo.Var(model.idxu)

model.obj = pyo.Objective(
    expr = 0.5* (model.x[0]**2 + model.x[1]**2 + model.u[0]**2 + model.u[1]**2),
    sense = pyo.minimize 
)

model.cons1 = pyo.Constraint(expr = model.x[0] == 0.5*x0+ model.u[0])
model.cons2 = pyo.Constraint(expr = model.x[1] == 0.5*model.x[0]+ model.u[1])
model.cons3 = pyo.Constraint(expr = (2,model.x[0],5))
model.cons4 = pyo.Constraint(expr = (-2,model.x[1],5))
model.cons5 = pyo.Constraint(expr = (-1,model.u[0],1) )
model.cons6 = pyo.Constraint(expr = (-1,model.u[1],1))

result = pyo.SolverFactory('ipopt').solve(model)

print('x* = ',[model.x[i]() for i in model.idxx])
print('opt_value = ', model.obj())
