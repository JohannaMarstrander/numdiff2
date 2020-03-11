# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:06:21 2020

@author: johan
"""

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
    ax.view_init(30, 110)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

class Problem2:
    def __init__(self, M, N, a, b, T, mu, g0, g1, u, f, params):
        self.M = M  # number of steps in space
        self.h = (b - a) / M  # step length in space
        self.a = a #Start domain space
        self.b = b #End domain space
        self.N = N  # number of steps in time
        self.k = T / N  # step length in space
        self.r = mu * self.k / (self.h * self.h)
        self.g0 = g0 #x=a
        self.g1 = g1 #x = b
        self.u = u #t = 0 (or exact solution)
        self.f = f
        self.t= np.linspace(0, T, N + 1)
        self.x = np.linspace(a, b, M+1)
        self.params = params

    def A_star(self):
        A = diags([-self.r / 2, 1 + self.r, -self.r / 2], [-1, 0, 1], shape=(self.M - 1, self.M - 1))
        #print(A.toarray())
        return A.tocsr()

    def F_star(self,U, other, u_star0, u_star1):
        # print(U)
        F = self.r / 2 * U[:-2] + (1 - self.r) * U[1:-1] + self.r / 2 * U[2:] + self.k * self.f(U[1:-1],other[1:-1], self.params)
        F[0] += self.r/2 * u_star0
        F[-1] += self.r/2 * u_star1
        return F


def solve(prob1, prob2):
    S = np.zeros((prob1.N + 1, prob1.M + 1))
    S[0, 1:-1] = prob1.u(prob1.x[1:-1])/(prob1.M + 1)
    S[:, 0] = prob1.g0( prob1.a)/(prob1.M + 1)
    S[:, -1] = prob1.g1(prob1.b)/(prob1.M + 1)
    
    I = np.zeros((prob2.N + 1, prob2.M + 1))
    I[0, 1: -1] = prob2.u(prob2.x[1:-1])/(prob2.M + 1)
    I[:, 0] = prob2.g0( prob2.a)/(prob2.M + 1)
    I[:, -1] = prob2.g1(prob2.b)/(prob2.M + 1)

    AS_star = prob1.A_star()
    AI_star = prob2.A_star()
    for n in range(prob1.N): #Assume they are the same for p1, p2, should organize differently
        ustarS0=(S[n + 1, 0] +(-prob1.params[0]*I[n,0])*prob1.k/2 * S[n,0])/(1+(-prob1.params[0]*I[n,0])*prob1.k/2)
        ustarS1 = (S[n + 1, -1] + (-prob1.params[0]*I[n,-1]) * prob1.k / 2 * S[n, -1])/(1+(-prob1.params[0]*I[n,-1])*prob1.k/2)
        ustarI0=(I[n + 1, 0] + (prob2.params[0]*S[n,0] -prob2.params[1])*prob2.k/2 * I[n,0])/(1+(prob2.params[0]*S[n,0] -prob2.params[1])*prob2.k/2)
        ustarI1 = (I[n + 1, -1] + (prob2.params[0]*S[n,-1] -prob2.params[1]) * prob2.k / 2 * I[n, -1])/(1+(prob2.params[0]*S[n,-1] -prob2.params[1])*prob2.k/2)
        FS_star = prob1.F_star(S[n],I[n],ustarS0,ustarS1)
        S_star = spsolve(AS_star, FS_star)
        FI_star = prob2.F_star(I[n], S[n],ustarI0,ustarI1)
        I_star = spsolve(AI_star, FI_star)
        S[n + 1, 1:-1] = S_star + prob1.k / 2 * (prob1.f(S_star, I_star, params) - prob1.f(S[n, 1:-1], I[n, 1:-1], params))
        I[n + 1, 1:-1] = I_star + prob1.k / 2 * (prob2.f(S_star, I_star, params) - prob2.f(S[n, 1:-1], I[n, 1:-1], params))
    return S, I


beta = 0.5
gamma = 1.0
mu_S = 0.001
mu_I = 0.001

S0 = 0.999
I0 = 0.001

params = [beta, gamma]
init = [S0,I0]

def fS(s1, s2, params):
    return (-params[0]*s1*s2)

def fI(s1, s2, params):
    return params[0]*s1*s2 - params[1]*s1

def gS(x):
    return 0*x + 1

def gI(x):
    return 0*x

def uS(x):
    return 1 + 0*x

def uI(x):
    return 0*x


s = Problem2(40, 180, 0, 1, 1, mu_S, gS, gS, uS, fS, params)
i = Problem2(40, 180, 0, 1, 1, mu_I, gI, gI, uI, fI, params)
US, UI = solve(s,i)

xx,tt=np.meshgrid(s.x,s.t)
plott(xx,tt,US + UI)
plott(xx,tt,UI)

