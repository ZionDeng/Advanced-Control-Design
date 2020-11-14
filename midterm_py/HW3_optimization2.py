# Linear and quadratic programming 
import pyomo.environ as pyo
import numpy as np
def check_solver_status(model, results):
    from pyomo.opt import SolverStatus, TerminationCondition
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print('========================================================================================')
        print('================ Problem is feasible and the optimal solution is found ==================')
        # print('z1 optimal=', pyo.value(model.z[1]))
        # print('z2 optimal=', pyo.value(model.z[2]))
        print('z1 optimal=', pyo.value(model.z1))
        print('z2 optimal=', pyo.value(model.z2))       
        print('optimal value=', pyo.value(model.obj))
        print('========================================================================================')
        bound = True
        feas = True
        # zOpt = np.array([pyo.value(model.z[1]), pyo.value(model.z[2])])
        zOpt = np.array([pyo.value(model.z1), pyo.value(model.z2)])

        JOpt = pyo.value(model.obj)
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print('========================================================')
        print('================ Problem is infeasible ==================')
        print('========================================================')
        feas = False
        zOpt = []
        JOpt = []
        if (results.solver.termination_condition == TerminationCondition.unbounded):
            print('================ Problem is unbounded ==================')
            bound = False
        else:
            bound = True
        
    else:
        if (results.solver.termination_condition == TerminationCondition.unbounded):
            print('================ Problem is unbounded ==================')
            bound = False
            feas = True
            zOpt = []
            JOpt = np.inf
        else:
            bound = True
            feas = True
            zOpt = []
            JOpt = np.inf
            
    return feas, bound, zOpt, JOpt

def LPQPa():
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var()
    m.z2 = pyo.Var()
    m.obj = pyo.Objective(expr = -5* m.z1-7* m.z2)
    m.cons1 = pyo.Constraint(expr = -3 * m.z1 + 2*m.z2 <= 30)
    m.cons2 = pyo.Constraint(expr = -2 * m.z1 + m.z2 <= 12)
    m.cons3 = pyo.Constraint(expr = m.z1 >=0)
    m.cons4 = pyo.Constraint(expr = m.z2 >=0)

    sol = pyo.SolverFactory('cbc').solve(m)
    return check_solver_status(m,sol)
# print(LPQPa())

def LPQPb():
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var()
    m.z2 = pyo.Var()
    m.obj = pyo.Objective(expr = 3*m.z1 + m.z2)

    m.cons1 = pyo.Constraint(expr = -1*m.z1 -1*m.z2 <=1)
    m.cons2 = pyo.Constraint(expr = 3*m.z1 + 2 *m.z2 <=12)
    m.cons3 = pyo.Constraint(expr = 2*m.z1 + 3*m.z2 <=3)
    m.cons4 = pyo.Constraint(expr = -2*m.z1 + 3*m.z2 >=9)
    m.cons5 = pyo.Constraint(expr = m.z1 >=0)
    m.cons6 = pyo.Constraint(expr = m.z2 >=0)

    res = pyo.SolverFactory('cbc').solve(m)
    return check_solver_status(m,res)

# print(LPQPb())

def LPQPc():
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var()
    m.z2 = pyo.Var()
    m.t11 = pyo.Var()
    m.t12 = pyo.Var()
    m.tinf = pyo.Var()
    m.obj = pyo.Objective(expr = m.t11 + m.t12 + m.tinf)

    m.cons1 = pyo.Constraint(expr = 3*m.z1 + 2*m.z2 <= -3)
    m.cons2 = pyo.Constraint(expr = (0,m.z1,2))
    m.cons3 = pyo.Constraint(expr = (-2,m.z2,3))

    m.cons4 = pyo.Constraint(expr = m.tinf >= m.z1 -2)
    m.cons5 = pyo.Constraint(expr = m.tinf >= -(m.z1 -2))
    m.cons6 = pyo.Constraint(expr = m.tinf >= m.z2)
    m.cons7 = pyo.Constraint(expr = m.tinf >= -m.z2)

    m.cons8 = pyo.Constraint(expr = m.t11 >= m.z1)
    m.cons9 = pyo.Constraint(expr = m.t11 >= -m.z1)
    m.cons10 = pyo.Constraint(expr = m.t12 >= m.z2+5)
    m.cons11 = pyo.Constraint(expr = m.t12 >= -(m.z2+5))

    res = pyo.SolverFactory('cbc').solve(m)
    return check_solver_status(m,res)

# print(LPQPc())

def LPQPd():
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var()
    m.z2 = pyo.Var()
    m.obj = pyo.Objective(expr = m.z1**2 + m.z2**2)
    m.cons1 = pyo.Constraint(expr = m.z1 <= -3)
    m.cons2 = pyo.Constraint(expr = m.z2<= 4)
    m.cons3 = pyo.Constraint(expr = 4*m.z1 + 3*m.z2 <=0)

    res = pyo.SolverFactory('ipopt').solve(m)
    return check_solver_status(m,res)

# print(LPQPd())

def NLP1(z0 = []):
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var(initialize = z0[0])
    m.z2 = pyo.Var(initialize = z0[1])
    m.obj = pyo.Objective(
        expr = 3*pyo.sin(-2*np.pi*m.z1) + 2*m.z1 +4+ pyo.cos(2*np.pi*m.z2)+ m.z2
    )

    m.cons1 = pyo.Constraint(expr = (-1,m.z1,1))
    m.cons2 = pyo.Constraint(expr = (-1,m.z2,1))
    res = pyo.SolverFactory('ipopt').solve(m)

    return [m.z1(),m.z2(),m.obj()]

# print(NLP1())

z1 = []
z2 = []
J = []
for _ in range(0):
    z1_init = np.random.uniform(-1,1)
    z2_init = np.random.uniform(-1,1)
    z1Opt,z2Opt,JOpt = NLP1([z1_init,z2_init])

    z1.append(z1Opt)
    z2.append(z2Opt)
    J.append(JOpt)

# print('z1Opt = ',z1)
# print('z2Opt = ',z2)
# print('JOpt = ',J)

z1_opt = z1
z2_opt = z2

import matplotlib.pyplot as plt 

if False:
    fig,ax = plt.subplots(figsize = (15,15))
    z = np.linspace(-1,1,100)
    z1_grid,z2_grid = np.meshgrid(z,z)
    C = 3* np.sin(-2* np.pi * z1_grid) + 2*z1_grid + 4+ np.cos(2* np.pi* z2_grid) + z2_grid
    contour = plt.contour(
        z1_grid,z2_grid,C,
        cmap=plt.cm.RdBu,
        vmin = abs(C).min(),vmax = abs(C).max(),
    )
    ax.clabel(contour,fontsize = 10,inline=1)
    ax.axis('square')
    ax.scatter(z1_opt,z2_opt,c ='b')
    plt.show()

if False:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12,9))
    ax = Axes3D(fig)
    z = np.linspace(-1,1,100)
    z1_grid,z2_grid = np.meshgrid(z,z)
    C = 3* np.sin(-2* np.pi * z1_grid) + 2*z1_grid + 4+ np.cos(2* np.pi* z2_grid) + z2_grid
    ax.plot_surface(z1_grid,z2_grid,C,rstride=1,cstride =1,cmap='viridis')
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('cost')
    plt.show()

# %% P3 NLP II

def NLP2(z0 = []):
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var(initialize = z0[0])
    m.z2 = pyo.Var(initialize = z0[1])
    m.obj = pyo.Objective(
        expr = pyo.log(1+m.z1**2) - m.z2
    )
    m.cons1 = pyo.Constraint(
        expr = -(1+m.z1**2)**2 + m.z2**2 ==4
    )
    res = pyo.SolverFactory('ipopt').solve(m)

    return [m.z1(),m.z2(),m.obj()]

# print(NLP2([0,0]))
z1 = []
z2 = []
J = []
for _ in range(0):
    z1_init = np.random.uniform(-1,1)
    # z2_init = np.random.uniform(-1,1)
    # -----------pay attention to z2_init here---------
    z2_init = np.sqrt( 4+ (1+z1_init**2)**2)
    z1Opt,z2Opt,JOpt = NLP2([z1_init,z2_init])

    z1.append(z1Opt)
    z2.append(z2Opt)
    J.append(JOpt)
# print('z1Opt = ',z1)
# print('z2Opt = ',z2)
# print('JOpt = ',J)

if False:
    fig,ax = plt.subplots(figsize = (15,15))
    z =np.linspace(-10,10,100)
    z1_grid, z2_grid = np.meshgrid(z,z)
    C = np.log(1 + z1_grid**2) - z2_grid
    CS = ax.contour(
        z1_grid,z2_grid,C,
        cmap = plt.cm.RdBu,
        vmin = abs(C).min(),vmax = abs(C).max()
    )
    ax.clabel(CS,fontsize=12,inline=1)
    ax.axis('square')
    plt.show()

from mpl_toolkits.mplot3d import Axes3D

if False:
    fig = plt.figure(figsize=(12,9))
    ax = Axes3D(fig)
    z = np.linspace(-10,10,100)
    z1_grid,z2_grid = np.meshgrid(z,z)
    C = np.log(1 + z1_grid**2) - z2_grid
    ax.plot_surface(z1_grid,z2_grid,C,rstride=1,cstride =1,cmap='viridis')
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('cost')
    plt.show()


# %% mixed integer problems 
def MIPa():
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var(within =pyo.Integers)
    m.z2 = pyo.Var(within =pyo.Integers)

    m.obj = pyo.Objective(
        expr = -6*m.z1 -5*m.z2
    )

    m.cons1 = pyo.Constraint(expr = 1* m.z1 + 4*m.z2 <=16)
    m.cons2 = pyo.Constraint(expr = 6* m.z1 + 4*m.z2 <=28)
    m.cons3 = pyo.Constraint(expr = 2* m.z1 - 5*m.z2 <=6)
    m.cons4 = pyo.Constraint(expr = (0,m.z1,10))
    m.cons5 = pyo.Constraint(expr = (0,m.z2,10))

    res = pyo.SolverFactory('glpk').solve(m)
    return [m.z1(),m.z2(),m.obj()]

# print(MIPa())

def MIPb():
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var()
    m.z2 = pyo.Var()
    m.bin = pyo.Var(within = pyo.Binary)
    m.obj = pyo.Objective(expr = -m.z1-2*m.z2)

    m.cons1 = pyo.Constraint(expr = 3*m.z1 + 4*m.z2 <=12+m.bin*1e8)
    m.cons2 = pyo.Constraint(expr = 4*m.z1 + 3*m.z2 <=12+(1-m.bin)*1e8)
    # -----------pay attention to the trick here------------------------
    m.c2 = pyo.Constraint(expr = m.z1>=0)
    m.c3 = pyo.Constraint(expr = m.z2>=0)
    res = pyo.SolverFactory('glpk').solve(m)
    return [m.z1(),m.z2(),m.obj()]

# print(MIPb())

# %% KKT conditions
from pyomo.opt import SolverStatus,TerminationCondition

def LPQPkkta():
    KKTsat = False
    Threshold = 1e-5

    m = pyo.ConcreteModel()
    m.z1 = pyo.Var()
    m.z2 = pyo.Var()
    m.obj = pyo.Objective(expr = m.z1**2 + m.z2**2)
    m.c1 = pyo.Constraint(expr = m.z1 <=-3)
    m.c2 = pyo.Constraint(expr = m.z2 <=4)
    m.c3 = pyo.Constraint(expr = 4*m.z1 + 3*m.z2 <=0)
    

    m.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)

    res = pyo.SolverFactory('ipopt').solve(m)
    if res.solver.termination_condition is not TerminationCondition.optimal:
        KKTsat = False
    else:
        zOpt = np.array([m.z1(),m.z2()])
        A = np.array([1,0,0,1,4,3]).reshape((3,2))
        b = np.array([-3,4,0])
        # ---------------------don't reshape here ---------------------

        u = []
        for c in m.component_objects(pyo.Constraint,active = True):
            print ("Constraint", c)
            for index in c:
                print(m.dual[c[index]])
                u.append(m.dual[c[index]])

        u = np.asarray(u)
        
        for i in range(len(u)):
            if u[i] < Threshold and u[i] > -Threshold:
                u[i] =0

        x = A @ zOpt - b
        flag_ineq = np.any(np.all(x <= Threshold) or np.all(x <= -Threshold))
        # flag_ineq = np.any(np.all(A @ zOpt <= b + Threshold) or np.all(A@zOpt <= b -Threshold))
        flag_dual = np.all(u>=0)
        # flag_cs = np.all(np.multiply(u,x) < Threshold and np.all(np.multiply(u,x) > -Threshold))
        flag_cs = np.all(np.multiply(u,x)< Threshold) and np.all(np.multiply(u,x)> -Threshold)
        # ------------------- pay attention here ---------------------

        grad_lagrangian = [2*zOpt[0],2*zOpt[1]] + u.T @A
        for i, y in enumerate(grad_lagrangian):
            if y < Threshold and y > -Threshold:
                grad_lagrangian[i] = 0

        flag_grad = np.all(grad_lagrangian ==0)
        flags = [flag_ineq,flag_dual,flag_cs,flag_grad]
        flags = np.array(flags)
        if all(flags ==1):
            KKTsat = True
        else:
            KKTsat = False 

        return KKTsat 

# print(LPQPkkta())

def NPLkkt(z0):
    KKTsat = True
    Threshold = 1e-5
    m = pyo.ConcreteModel()
    m.z1 = pyo.Var(initialize = z0[0])
    m.z2 = pyo.Var(initialize = z0[1])
    m.obj = pyo.Objective(
        expr = 3*pyo.sin(-2*np.pi*m.z1) + 2*m.z1+4+pyo.cos(2*np.pi*m.z2) + m.z2
    )
    # m.c1 = pyo.Constraint(expr = (-1,m.z1,1))
    # m.c2 = pyo.Constraint(expr = (-1,m.z2,1))
    # ------------------this constraint should only be expressed like this--------------------
    m.c11 = pyo.Constraint(expr = m.z1 <=1)
    m.c12 = pyo.Constraint(expr = -m.z1 <=1)
    m.c21 = pyo.Constraint(expr = m.z2 <=1)
    m.c22 = pyo.Constraint(expr = -m.z2 <=1)


    m.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT)

    res = pyo.SolverFactory('ipopt').solve(m)
    zOpt = [m.z1(),m.z2()]

    if res.solver.termination_condition is not TerminationCondition.optimal:
        return False 

    # A = np.array([1,0,-1,0,0,1,0,-1]).reshape((4,2))
    A = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    b = np.array([1,1,1,1])

    u = []
    for c in m.component_objects(pyo.Constraint,active = True):
        for i in c:
            u.append(m.dual[c[i]])

    for i,ui in enumerate(u):
        if ui < Threshold and ui> -Threshold:
            u[i] = 0

    u = np.asarray(u)

    x = A @ zOpt -b 
    flag_ineq = np.all(x <= Threshold)
    flag_dual = np.all(u >= 0)
    flag_cs = np.all(np.multiply(u,x)<Threshold) and np.all(np.multiply(u,x)>-Threshold)
    grad_lagrangian = [ -6*np.pi * np.cos(-2*np.pi*zOpt[0]) +2,-2*np.pi*np.sin(2*np.pi*zOpt[1])+1] + u.T @A

    for i, y in enumerate(grad_lagrangian):
        if y < Threshold and y > -Threshold:
            grad_lagrangian[i] = 0

    flag_grad = np.all(grad_lagrangian ==0)
    flags = [flag_ineq,flag_dual,flag_cs,flag_grad]
    flags = np.array(flags)
    if all(flags ==1):
        KKTsat = True
    else:
        KKTsat = False 

    return KKTsat

print(NPLkkt([0,0]))



    