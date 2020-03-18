from scipy import sparse

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm
import matplotlib.animation as animation


def _plot_frame_fdm_solution(i, ax, X, Y, U_list, title, zlim=None):
    ax.clear()
    line = ax.plot_surface(X, Y, U_list[i],
                           rstride=1, cstride=1,  # Sampling rates for the x and y input data
                           cmap=cm.viridis)  # Use the new fancy colormap viridis
    if zlim is not None:
        ax.set_zlim(zlim)
    total_frame_number = len(U_list)
    complete_title = title + (" (Frame %d of %d)" % (i, total_frame_number))
    ax.set_title(complete_title)
    return line


def plot_2D_animation(X, Y, U_list, title='', duration=10, zlim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fargs = (ax, X, Y, U_list, title, zlim)
    frame_plotter = _plot_frame_fdm_solution

    frames = len(U_list)
    interval = duration / frames * 1000
    ani = animation.FuncAnimation(fig, frame_plotter,
                                  frames=len(U_list), fargs=fargs,
                                  interval=interval, blit=False, repeat=True)
    return ani


def I_map(i, j, n):
    """Define index mapping"""
    return i + j * (n + 1)


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


class Problem2:
    def __init__(self, M, N, x0, xM, T, muS, muI, St0, It0, fS, fI, beta, gamma):
        self.M = M  # number of steps in space
        self.h = (xM - x0) / M  # step length in space
        self.N = N  # number of steps in time
        self.k = T / N  # step length in time
        self.rS = muS * self.k / (self.h * self.h)
        self.rI = muI * self.k / (self.h * self.h)
        self.St0 = St0  # t = 0
        self.It0 = It0  # t = 0
        self.fS = fS
        self.fI = fI
        self.beta = beta
        self.gamma = gamma
        self.t = np.linspace(0, T, N + 1)
        self.x = np.linspace(x0, xM, M + 1)


def solve(p, A_star, F_star):
    S, I = np.zeros((p.N + 1, (p.M + 1) ** 2)), np.zeros((p.N + 1, (p.M + 1) ** 2))
    Sm, Im = np.zeros((p.N + 1, p.M + 1, p.M + 1)), np.zeros((p.N + 1, p.M + 1, p.M + 1))
    S[0], I[0] = p.St0(p.x), p.It0(p.x)  # t = 0
    Sm[0], Im[0] = S[0].reshape(p.M + 1, p.M + 1), I[0].reshape(p.M + 1, p.M + 1)
    SA_star = A_star(p.rS, p.M)
    IA_star = A_star(p.rI, p.M)
    for n in tqdm(range(p.N)):
        SF_star = F_star(p.rS, p.k, p.M, S[n], p.fS, p.beta, p.gamma, I[n])
        IF_star = F_star(p.rI, p.k, p.M, I[n], p.fI, p.beta, p.gamma, S[n])
        SU_star = spsolve(SA_star, SF_star)
        IU_star = spsolve(IA_star, IF_star)
        S[n + 1, :] = SU_star + p.k / 2 * (
                p.fS(SU_star, I[n],  p.beta, p.gamma) - p.fS(S[n], I[n], p.beta, p.gamma))
        I[n + 1, :] = IU_star + p.k / 2 * (
                p.fI(IU_star, S[n], p.beta, p.gamma) - p.fI(I[n], S[n], p.beta, p.gamma))
        Sm[n + 1, :] = S[n + 1, :].reshape(p.M + 1, p.M + 1)
        Im[n + 1, :] = I[n + 1, :].reshape(p.M + 1, p.M + 1)
    return Sm, Im


def A_star(r, M):
    A = np.zeros(((M + 1) ** 2, (M + 1) ** 2))
    np.fill_diagonal(A, 1 + 2* r)
    for i in range(1, M):
        for j in range(1, M):
            A[I_map(i, j, M), I_map(i - 1, j, M)] = -r / 2
            A[I_map(i, j, M), I_map(i + 1, j, M)] = -r / 2
            A[I_map(i, j, M), I_map(i, j - 1, M)] = -r / 2
            A[I_map(i, j, M), I_map(i, j + 1, M)] = -r / 2

    for j in range(1, M):
        A[I_map(0, j, M), I_map(1, j, M)] = -r
        A[I_map(0, j, M), I_map(0, j + 1, M)] = -r / 2
        A[I_map(0, j, M), I_map(0, j - 1, M)] = -r / 2
        A[I_map(M, j, M), I_map(M - 1, j, M)] = -r
        A[I_map(M, j, M), I_map(M, j + 1, M)] = -r / 2
        A[I_map(M, j, M), I_map(M, j - 1, M)] = -r / 2

    for i in range(1, M):
        A[I_map(i, 0, M), I_map(i, 1, M)] = -r
        A[I_map(i, 0, M), I_map(i + 1, 0, M)] = -r / 2
        A[I_map(i, 0, M), I_map(i - 1, 0, M)] = -r / 2
        A[I_map(i, M, M), I_map(i, M - 1, M)] = -r
        A[I_map(i, M, M), I_map(i + 1, M, M)] = -r / 2
        A[I_map(i, M, M), I_map(i - 1, M, M)] = -r / 2

    A[I_map(0, 0, M), I_map(0, 1, M)] = -r
    A[I_map(0, 0, M), I_map(1, 0, M)] = -r

    A[I_map(0, M, M), I_map(0, M - 1, M)] = -r
    A[I_map(0, M, M), I_map(1, M, M)] = -r

    A[I_map(M, 0, M), I_map(M - 1, 0, M)] = -r
    A[I_map(M, 0, M), I_map(M, 1, M)] = -r

    A[I_map(M, M, M), I_map(M - 1, M, M)] = -r
    A[I_map(M, M, M), I_map(M, M - 1, M)] = -r

    return sparse.csr_matrix(A)


def F_star(r, k, M, U, f, beta, gamma, other):
    F = np.zeros((M + 1) ** 2)
    for i in range(1, M):
        for j in range(1, M):
            F[I_map(i, j, M)] = r / 2 * U[I_map(i - 1, j, M)] + r / 2 * U[I_map(i + 1, j, M)] + r / 2 * U[
                I_map(i, j - 1, M)] + r / 2 * U[I_map(i, j + 1, M)] + (1 - 2 * r) * U[I_map(i, j, M)] + k * f(
                U[I_map(i, j, M)],
                other[I_map(i, j, M)], beta[I_map(i, j, M)], gamma)

    for j in range(1, M):
        F[I_map(0, j, M)] = r * U[I_map(1, j, M)] + r / 2 * U[I_map(0, j - 1, M)] + r / 2 * U[I_map(0, j + 1, M)] \
                            + (1 - 2 * r) * U[I_map(0, j, M)] + k * f(U[I_map(0, j, M)], other[I_map(0, j, M)],
                                                                      beta[I_map(0, j, M)], gamma)
        F[I_map(M, j, M)] = r * U[I_map(M - 1, j, M)] + r / 2 * U[I_map(M, j - 1, M)] + r / 2 * U[I_map(M, j + 1, M)] \
                            + (1 - 2 * r) * U[I_map(M, j, M)] + k * f(U[I_map(M, j, M)], other[I_map(M, j, M)],
                                                                      beta[I_map(M, j, M)], gamma)

    for i in range(1, M):
        F[I_map(i, 0, M)] = r * U[I_map(i, 1, M)] + r / 2 * U[I_map(i - 1, 0, M)] + r / 2 * U[I_map(i + 1, 0, M)] \
                            + (1 - 2 * r) * U[I_map(i, 0, M)] + k * f(U[I_map(i, 0, M)], other[I_map(i, 0, M)],
                                                                      beta[I_map(i, 0, M)], gamma)
        F[I_map(i, M, M)] = r * U[I_map(i, M - 1, M)] + r / 2 * U[I_map(i - 1, M - 1, M)] + r / 2 * U[
            I_map(i + 1, M - 1, M)] \
                            + (1 - 2 * r) * U[I_map(i, M, M)] + k * f(U[I_map(i, M, M)], other[I_map(i, M, M)],
                                                                      beta[I_map(i, M, M)], gamma)

    F[I_map(0, 0, M)] = r * U[I_map(1, 0, M)] + r * U[I_map(0, 1, M)] + (1 - 2 * r) * U[I_map(0, 0, M)] \
                        + k * f(U[I_map(0, 0, M)], other[I_map(0, 0, M)], beta[I_map(0, 0, M)], gamma)
    F[I_map(M, 0, M)] = r * U[I_map(M - 1, 0, M)] + r * U[I_map(M, M - 1, M)] + (1 - 2 * r) * U[I_map(M, 0, M)] \
                        + k * f(U[I_map(M, 0, M)], other[I_map(M, 0, M)], beta[I_map(M, 0, M)], gamma)
    F[I_map(0, M, M)] = r * U[I_map(0, M - 1, M)] + r * U[I_map(1, M, M)] + (1 - 2 * r) * U[I_map(0, M, M)] \
                        + k * f(U[I_map(0, M, M)], other[I_map(0, M, M)], beta[I_map(0, M, M)], gamma)
    F[I_map(M, M, M)] = r * U[I_map(M, M - 1, M)] + r * U[I_map(M - 1, M, M)] + (1 - 2 * r) * U[I_map(M, M, M)] \
                        + k * f(U[I_map(M, M, M)], other[I_map(M, M, M)], beta[I_map(M, M, M)], gamma)

    return F


beta = 3.0
gamma = 1.0
mu_S = 0.001
mu_I = 0.001

params = [beta, gamma]


def St0(x):
    M = len(x)
    t0 = np.ones((M) ** 2)
    t0[M ** 2 // 2] = 0
    # print(t0)
    return t0


def It0(x):
    M = len(x)
    t0 = np.zeros((M) ** 2)
    t0[M ** 2 // 2] = 1
    # print(t0)
    return t0


def fS(S, I, beta,gamma):
    return (-beta * S*I)


def fI(I, S, beta,gamma):
    return beta *S* I - gamma * I

beta=3*np.ones((40+1)**2)
#beta[(40+1)**2//3] = 10
#beta=2

prob = Problem2(40, 100, 0, 1, 10, mu_S, mu_I, St0, It0, fS, fI, beta, gamma)
S1, I1 = solve(prob, A_star, F_star)

xx, yy = np.meshgrid(prob.x, prob.x)
plott(xx, yy, I1[0])
plott(xx, yy, I1[1])
plott(xx, yy, I1[-1])
plot_2D_animation(xx, yy, I1, title='', duration=10, zlim=(-1, 1))
