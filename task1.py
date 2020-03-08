import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



class Problem:
    def __init__(self, M, N, a, b, T, mu, g0, g1, u, f):
        self.M = M  # number of steps in space
        self.h = (b - a) / M  # step length in space
        self.a = a
        self.b = b
        self.N = N  # number of steps in time
        self.k = T / N  # step length in space
        self.r = mu * self.k / (self.h * self.h)
        self.g0 = g0
        self.g1 = g1
        self.u = u
        self.f = f
        self.t= np.linspace(0, T, N + 1)
        self.x = np.linspace(a, b, M+1)

    def A_star(self):
        A = diags([-self.r / 2, 1 + self.r, -self.r / 2], [-1, 0, 1], shape=(self.M - 1, self.M - 1))
        #print(A.toarray())
        return A.tocsr()

    def F_star(self, U, u_star0, u_star1):
        # print(U)
        F = self.r / 2 * U[:-2] + (1 - self.r) * U[1:-1] + self.r / 2 * U[2:] + self.k * f(U[1:-1])
        F[0] += self.r/2 * u_star0
        F[-1] += self.r/2 * u_star1
        return F


def solve(prob):
    U = np.zeros((prob.N + 1, prob.M + 1))
    U[0] = prob.u(0, prob.x)
    U[:, 0] = prob.g0(prob.t, prob.a)
    U[:, -1] = prob.g1(prob.t, prob.b)

    A_star = prob.A_star()
    for n in range(prob.N):
        F_star = prob.F_star(U[n],U[n + 1, 0],U[n + 1, -1])
        U_star = spsolve(A_star, F_star)
        U[n + 1, 1:-1] = U_star + prob.k / 2 * (f(U_star) - f(U[n, 1:-1]))
    return U


def f(x):
    return 0.5 * x


def u(t, x):
    return np.exp(2*x + t)





a = Problem(200, 50, 0, 1, 1, 0.125, u, u, u, f)
#b=Problem(5, 400, 0, 1, 1, 0.1, u, u, u, f)
U1 = solve(a)
#U2= solve(b)
#print(U)


xx,tt=np.meshgrid(a.x,a.t)
#print(u(tt,xx))

def plott(x, y, Z):
    """Function for 3D plotting,
    edited from https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(30, 110)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

#plott(xx,tt,U1-U2)
#print(xx.shape,tt.shape,U.shape)
plott(xx,tt,U1)
u_ex=u(tt,xx)
plott(xx,tt,u_ex)
error=U1-u_ex
plott(xx,tt,error)

print(np.max(np.absolute(error)))

