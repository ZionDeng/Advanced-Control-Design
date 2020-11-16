# scipy.intergrate.solve_ivp 
from scipy.integrate import solve_ivp
import numpy as np 
import matplotlib.pyplot as plt 

# %% solve differential equation 
# x0 = 1, t0 = 0, tf = 20 

def exp_ode(t,x):
    return np.exp(-x)

x0 = np.array([1])
t0 = 0 
tf = 20 

# sol = solve_ivp(exp_ode,[t0,tf], x0) 
sol = solve_ivp(lambda t,x: np.exp(-x),[t0,tf],x0)
# plt.plot(sol.t,sol.y[0])
# plt.show()

# %% 

sol = solve_ivp(lambda t,x: -x * np.exp(t),[0,5],[1])
# plt.plot(sol.t,sol.y[0])
# plt.show()

# %% 
sol = solve_ivp(lambda t,x: (-0.2 + np.sin(t)) * x,[0,5],[5],'RK45',rtol = 1e-6)
# plt.plot(sol.t,sol.y[0])
# plt.show()


# %% 
a0, a1, b0, tf = -2.5, 0.05, 0.25, 15 

x0 = [np.pi/4, 0]

def dyn_ode(t,x):
    theta, thetadot = x
    return [thetadot, -a0 *np.sin(theta) - a1* thetadot + b0* np.cos(theta)]
# ---------------WHERE IS U HERE--------------------------??

sol = solve_ivp(dyn_ode,[t0,tf],x0,rtol= 1e-7)

# plt.plot(sol.t,sol.y[0,:])
# plt.plot(sol.t,sol.y[1,:])
# plt.show()

# %% analytical differentiation 
import sympy as sym 

x = sym.Symbol('x')
A = sym.Matrix(np.eye(4) * x)
# print(A)

A_val = A.subs(x,3)
# print(A_val)

x, y, z = sym.symbols('x y z')
Jacob = sym.Matrix(
    [sym.cos(y) + x, sym.sin(x) + y, z]
).jacobian([x,y,z])
# print(Jacob)

Jacob_val = Jacob.subs([
    (x,0), (y, np.pi/4)
])
print(Jacob_val)