# %% P1 

from numpy import sin,cos,tan,arcsin 
import numpy as np 
import sympy as sym 
import matplotlib.pyplot as plt 

m,l,g,a,k,d,beta = 2,6,9.81,3,1.5,1,0.1

def EquilPoint(theta2_bar, alpha_bar):
    Fsd_bar = (m * g* l *sin(theta2_bar) +alpha_bar) / (a * cos(theta2_bar))
    theta1_bar = arcsin(sin(theta2_bar) - Fsd_bar/(k*a))
    T_bar = - (m*g*l * sin(theta1_bar) + a*cos(theta1_bar) * Fsd_bar)
    return theta1_bar, T_bar

# print(EquilPoint(0.1,0.5))

# %% 
theta1dot, theta2dot = 0,0
def LinearizeModel(theta2,alpha):
    theta1, T = EquilPoint(theta2,alpha)

    x1, x2, x3, x4, u1, u2 = sym.symbols('x1 x2 x3 x4 u1 u2')
    f = sym.Matrix(
        [x2, 
        (m*g*l * sym.sin(x1) + a**2*sym.cos(x1) *(k*(sym.sin(x3) -sym.sin(x1))+ d*(x4-x2)) + u1)/ (m*l**2),
        x4,
        (m*g*l*sym.sin(x3) - a**2*sym.cos(x3)*(k*(sym.sin(x3)-sym.sin(x1))+d*(x4-x2))+u2+beta*x4**2)/(m*l**2)]
    )
    A = f.jacobian([x1,x2,x3,x4,u1,u2]).subs([
        (x1,theta1),(x2,theta1dot),(x3,theta2),(x4,theta2dot),(u1,T),(u2,alpha)
    ])

    # WHY DOES A RELATED TO U1,U2????
    B = f.jacobian([u1,u2]).subs([
        (x1,theta1),(x2,theta1dot),(x3,theta2),(x4,theta2dot),(u1,T),(u2,alpha)
    ])
    C = [1,0,0,0]
    D = [0,0]
    return A,B,C,D,theta1,T

theta2_bar = 0.1
alpha_bar = 0.5
# print(LinearizeModel(theta2_bar,alpha_bar))

#  %% 
def bldgHTM(T, u1,u2,q,mz,cz,cp):
    Tdot = (q + cp*u1*(u2-T))/(mz*cz)

mz, cz, Ts, cp = 100,20,0.1,1000

def eulerDiscretization(T,q,u1,u2):
    T_KplusOne = (1-cp*Ts/(mz*cz)*u1) *T + Ts/(mz*cz)*(q+cp*u1*u2)
    return T_KplusOne

# %% P3 
lr = 1.738
dt = 0.1 
def carModel(beta,a,x,y,psi,v):
    x_dot = v*cos(psi+beta)
    y_dot = v*sin(psi+beta)
    psi_dot = v/lr*sin(beta)
    v_dot = a 
    return x+x_dot*dt, y+y_dot*dt,psi+psi_dot*dt,v+v_dot*dt

# print(carModel(.1, 2, 5, 2,10,0.1))
from scipy.io import loadmat 
Data = loadmat('Data/sineData.mat')
a = Data['a']
beta = Data['beta']
time = Data['time']
Ts = 0.1 
Num_steps = len(time)

def sim(time,a,beta,x0):
    
    x,y,psi,v = x0 
    # initialize the arrays
    xtrend, ytrend, psi_trend,v_trend = [x],[y],[psi],[v]
    # use the function to iterate 
    for i in range(Num_steps-1):
        x,y,psi,v = carModel(beta[i],a[i],x,y,psi,v)
        xtrend.append(x)
        ytrend.append(y)
        psi_trend.append(psi)
        v_trend.append(v)

    # return np.asarray(xtrend),np.asarray(ytrend),np.asarray(psi_trend),np.asarray(v_trend)
    return xtrend,ytrend,psi_trend,v_trend
# call the function 
x,y,psi,v = 0,0,0,0
# x0 = np.array([x,y,psi,v])
x0 = [x,y,psi,v]
xtrend,ytrend,psi_trend,v_trend = sim(time,a,beta,x0)

# %% plot 

if False:
    N_plot = 200
    plt.subplot(411)
    plt.plot(time,xtrend)
    plt.ylabel('x')
    plt.subplot(412)
    plt.plot(time,ytrend)
    plt.ylabel('y')

    plt.subplot(413)
    plt.plot(time,psi_trend)
    plt.ylabel('psi')
    plt.subplot(414)
    plt.plot(time,v_trend)
    plt.ylabel('speed')

    # plt.show()

    plt.plot(xtrend,ytrend)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Path of Car')
    # plt.show()


