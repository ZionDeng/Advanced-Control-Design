import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from scipy.linalg import inv
from scipy import linalg

import scipy.signal
# import scipy.linalg

Ts = .1
a0 = -2.5
a1 = .05
b0 = .25
Ac = np.array([0, 1, -a0, -a1]).reshape((2, 2))
Bc = np.array([0, b0]).reshape((2, 1))

Cc = np.zeros((1, 2))
Dc = np.zeros((1, 1))

Ad, Bd, Cd, Dd, dt = scipy.signal.cont2discrete((Ac, Bc, Cc, Dc
                                                 ), Ts)

Q = np.diag([1, 0])
R = 1


def dlqr(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = inv(B.T @ P @ B + R) @ (B.T@P @ A)
    eigVals, eigVecs = linalg.eig(A - B @ K)
    return K, P, eigVals


Klqr, P, eigvals = dlqr(Ad, Bd, Q, R)


x0 = np.array([np.pi/4, 0])
Tf = 10
T0 = 0


# def continuous_dyn(t, x, u):
#     theta, theta_dot = x
#     return [theta_dot, -a0*np.sin(theta) - a1*theta_dot + b0*np.cos(theta)*u]


def continuous_dyn_with_controller(t, x, klqr):
    theta, theta_dot = x
    k1, k2 = klqr
    u = -k1 * theta - k2*theta_dot
    return [theta_dot, -a0 * np.sin(theta) - a1*theta_dot + b0*u*np.cos(theta)]


sol = solve_ivp(
    continuous_dyn_with_controller, [T0, Tf], x0,
    t_eval=np.arange(0, Tf, Ts),
    args=(Klqr)
    # it is not flatten here!!----------
)
u = [-Klqr @ sol.y[:, i] for i in range(len(sol.y[0, :]))]


plt.subplot(3, 1, 1)
plt.plot(sol.y[0, :])
plt.ylabel('theta')
plt.subplot(312)
plt.plot(sol.y[1, :])
plt.ylabel('theta_dot')
plt.subplot(313)
plt.plot(u, 'r')
plt.ylabel('u')
plt.show()
