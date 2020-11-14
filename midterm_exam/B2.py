import numpy as np 
import cvxopt


def reg1Inf(A_in,b_in,c_in,d_in):
    # the question transformed to:
    # min(x1,x2,x3,t){t}
    # t = max(|A_in *x - b_in|)

    n_1, nx = np.shape(A_in)

    c = np.array([0]*nx + [1])
    A = np.concatenate(
        [
            np.concatenate([A_in,-np.ones((2,1))],axis=1),
            np.concatenate([-A_in,-np.ones((2,1))],axis=1),
            np.concatenate([c_in,np.zeros((2,1))],axis= 1)
        ],axis = 0
    )

    b = np.concatenate([b_in,-b_in,d_in],axis=0)

    c = cvxopt.matrix(c,tc='d')
    A = cvxopt.matrix(A,tc='d')
    b = cvxopt.matrix(b,tc='d')


    sol = cvxopt.solvers.lp(c,A,b)
    xOpt = sol['x']
    J = sol['primal objective']
    return xOpt[:nx],J 


a = np.array([[2,0,-1],[1,-1,0]])
b = np.array([1,2]).reshape((2,1))
c = np.array([[1,-1,1],[-1,-1,0]])
d = np.array([-1,-1]).reshape((2,1))
xOpt, J = reg1Inf(a,b,c,d)
print('xOpt: ',xOpt)
print('J* = ',J)