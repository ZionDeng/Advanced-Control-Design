# KKT conditions 

# %% quadratic programming example 

import numpy as np 
import pyomo.environ as pyo 
from pyomo.opt import TerminationCondition

Threshold = 1e-5 

model = pyo.ConcreteModel()
model.z1 = pyo.Var()
model.z2 = pyo.Var()

model.obj = pyo.Objective(expr = model.z1 **2 + model.z2**2)
# model.cons1 = pyo.Constraint(expr = 1 <= model.z1)
# model.cons2 = pyo.Constraint(expr = 1 <= model.z2)
model.cons1 = pyo.Constraint(expr = -model.z1 <= -1)
model.cons2 = pyo.Constraint(expr = -model.z2 <= -1)
# ---------------------only in the <= form ----------------------------------


model.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)

result = pyo.SolverFactory('ipopt').solve(model)
print('dual1 = ',model.dual[model.cons1])
print('dual2 = ',model.dual[model.cons2])
print('z1*_solver = ',model.z1())
print('z2*_solver = ',model.z2())
print('opt_value = ',model.obj())

# %% check the status 
if result.solver.termination_condition is not TerminationCondition.optimal:
    KKTsat = False 
else:
    A = -np.eye(2)
    b= -np.ones((2,1))
    zOpt = np.array([model.z1(),model.z2()])
    u = []
    for c in model.component_objects(pyo.Constraint,active = True):
        print('Constraint',c)
        for i in c:
            u.append(model.dual[c[i]])
# -------------pay attention to here !!!------------
            print(model.dual[c[i]])

    u = np.asarray(u)
    for i in range(len(u)):
        if u[i] < Threshold and u[i] > Threshold:
            u[i] = 0
    
    flag_primal = np.any(
        np.all(A @ zOpt <= b + Threshold) or 
        np.all(A @ zOpt <= b - Threshold)  
    )

    flag_dual = np.all(u >=0 )

    flag_cs = np.all(np.multiply(u,(A@zOpt-b)) < Threshold)  and np.all(np.multiply(u,(A@zOpt-b)) > -Threshold)

    grad_lagrangian = [2*zOpt[0],2*zOpt[1]] + u.T @ A 

    for i in range(len(grad_lagrangian)):
        if grad_lagrangian[i] < Threshold and grad_lagrangian[i] > -Threshold:
            grad_lagrangian[i] = 0 
    flag_grad = np.all(grad_lagrangian ==0)

    flags = [flag_primal,flag_dual,flag_cs,flag_grad]
    if np.all(np.array(flags)==1):
        KKTsat = True
    else:
        KKTsat = False 

print(KKTsat)