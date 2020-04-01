import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    ax.view_init(20, 92)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


class Problem2:
    def __init__(self, M, N, x0, xM, T, muS, muI, St0, It0, Sx0, SxM, Ix0, IxM, fS, fI, params, tot):
        self.M = M  # number of steps in space
        self.h = (xM - x0) / M  # step length in space
        self.x0 = x0  # Start domain space
        self.xM = xM  # End domain space
        self.N = N  # number of steps in time
        self.k = T / N  # step length in time
        self.rS = muS * self.k / (self.h * self.h)
        self.rI = muI * self.k / (self.h * self.h)
        self.St0 = St0  # t = 0
        self.Sx0 = Sx0  # x = 0
        self.SxM = SxM  # x = M
        self.It0 = It0  # t = 0
        self.Ix0 = Ix0  # x = 0
        self.IxM = IxM  # x = M
        self.fS = fS
        self.fI = fI
        self.tot = tot
        self.params = params
        self.t = np.linspace(0, T, N + 1)
        self.x = np.linspace(x0, xM, M + 1)


def solve(p, A_star, F_star):
    # Creating grid for S and I and setting BC/IV
    S, I = np.zeros((p.N + 1, p.M + 1)), np.zeros((p.N + 1, p.M + 1))  # np.zeros((p.N + 1, p.M + 1))
    S[0], I[0] = p.St0(p.t, p.x), p.It0(p.t, p.x)  # t = 0
    S[:, 0], I[:, 0] = p.Sx0(p.t, p.x0), p.Ix0(p.t, p.x0)  # x = 0
    S[:, -1], I[:, -1] = p.SxM(p.t, p.xM), p.IxM(p.t, p.xM)  # x = M

    SA_star = A_star(p.rS, p.M)
    IA_star = A_star(p.rI, p.M)
    for n in range(p.N):
        SF_star = F_star(p.rS, p.k, S[n],  p.fS, p.params, I[n])
        IF_star = F_star(p.rI, p.k, I[n], p.fI, p.params, S[n])
        SU_star = spsolve(SA_star, SF_star)
        IU_star = spsolve(IA_star, IF_star)
        S[n + 1, :] = SU_star + p.k / 2 * (
                    p.fS(SU_star, I[n], p.params) - p.fS(S[n], I[n], p.params))
        I[n + 1, :] = IU_star + p.k / 2 * (
                    p.fI(IU_star, S[n], p.params) - p.fI(I[n], S[n], p.params))

    return S, I


def A_star(r, M):
    A = diags([-r / 2, 1 + r, -r / 2], [-1, 0, 1], shape=(M + 1, M + 1))
    A = A.tocsr()
    A[0,1]= -r
    A[-1,-2]=-r
    return A


def F_star(r, k, U, f, params, other):
    F = np.zeros(len(U))
    F[1:-1] = r / 2 * U[:-2] + (1 - r) * U[1:-1] + r / 2 * U[2:] + k * f(U[1:-1], other[1:-1], params)
    F[0] =(1 - r) * U[0] +r  * U[0]+ k * f(U[0], other[0], params)
    F[-1] = (1 - r) * U[-1] +r  * U[-1]+ k * f(U[-1], other[-1], params)
    return F


beta = 0.5
gamma = 1.0
mu_S = 0.005
mu_I = 0.005
tot = 0

params = [beta, gamma]


def St0(t, x):
    M = len(x)
    t0 = np.ones(M)
    t0[M // 2] = 0
    return t0


def It0(t, x):
    M = len(x)
    t0 = np.zeros(M)
    t0[M // 2] = 1
    return t0


def Sx0(t, x):
    N = len(t)
    x0 = np.ones(N)
    return x0


def Ix0(t, x):
    N = len(t)
    x0 = np.zeros(N)
    return x0


def fS(S, I, params):
    return (-params[0] * np.multiply(S, I))


def fI(I, S, params):
    return params[0] * np.multiply(S, I) - params[1] * I



prob = Problem2(50, 300, 0, 1, 4, mu_S, mu_I, St0, It0, Sx0, Sx0, Ix0, Ix0, fS, fI, params, tot)
S1, I1 = solve(prob, A_star, F_star)

xx, tt = np.meshgrid(prob.x, prob.t)
plott(xx, tt, I1)




