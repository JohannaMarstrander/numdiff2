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
        F = self.r / 2 * U[:-2] + (1 - self.r) * U[1:-1] + self.r / 2 * U[2:] + self.k * self.f(U[1:-1])
        F[0] += self.r/2 * u_star0
        F[-1] += self.r/2 * u_star1
        return F


def solve(prob,a):
    U = np.zeros((prob.N + 1, prob.M + 1))
    U[0] = prob.u(0, prob.x)
    U[:, 0] = prob.g0(prob.t, prob.a)
    U[:, -1] = prob.g1(prob.t, prob.b)

    A_star = prob.A_star()
    for n in range(prob.N):
        ustar0=(U[n + 1, 0] + a*prob.k/2 * U[n,0])/(1+a*prob.k/2)
        ustar1 = (U[n + 1, -1] + a * prob.k / 2 * U[n, -1])/(1+a*prob.k/2)
        F_star = prob.F_star(U[n],ustar0,ustar1)
        U_star = spsolve(A_star, F_star)
        U[n + 1, 1:-1] = U_star + prob.k / 2 * (f(U_star) - f(U[n, 1:-1]))
    return U


def f(x):
    return 0.5 * x

def u(t, x):
    return np.exp(2*x + t)


#Convergence order time
N_list = [10, 20, 35, 60, 100]
#N_list = [10,20,40,80, 120]
E_time =[]
k_list = []

for n in N_list:
    b = Problem(400, n, 0, 1, 1, 0.125, u, u, u, f)
    U1 = solve(b,0.5)
    xx,tt=np.meshgrid(b.x,b.t)
    u_ex=u(tt,xx)
    error = U1[-1,:] - u_ex[-1,:]
    if n == 60:
        plott(xx,tt,u_ex)
    E_time.append(np.linalg.norm(error, ord=np.inf))
    k_list.append(1 / n)
    print(np.max(np.absolute(error)))
    
print(E_time)
order = np.polyfit(np.log(k_list), np.log(E_time), 1)[0]
print("order time", order)
k_list = np.array(k_list)



plt.figure()
plt.loglog(k_list, E_time, 'o-', label = "Order = 2.08")
plt.loglog(k_list, k_list**2, 'r-')
plt.title("Time convergence error")
plt.xlabel("k")
plt.ylabel("Error")
plt.legend(loc = 'upper left')
plt.show()

#Convergence order space
M_list = [10, 20, 35, 60, 100]
#M_list = [20,40,80,160,320]
E_space =[]
h_list = []

for m in M_list:
    b = Problem(m, 400, 0, 1, 1, 0.125, u, u, u, f)
    U1 = solve(b,0.5)
    xx,tt=np.meshgrid(b.x,b.t)
    u_ex=u(tt,xx)
    error = U1[-1,:] -u_ex[-1,:]
    #if m == 80:
        #plott(xx,tt,u_ex)
    E_space.append(np.linalg.norm(error, ord=np.inf))
    h_list.append(1 / m)
    
order = np.polyfit(np.log(h_list), np.log(E_space), 1)[0]
print("order space", order)
h_list = np.array(h_list)

plt.figure()
plt.loglog(h_list, E_space, 'o-', label = "Order = 2.00")
plt.loglog(h_list, h_list**2, 'r-')
plt.title("Space convergence error")
plt.xlabel("h")
plt.ylabel("Error")
plt.legend(loc = 'upper left')
plt.show()

#print(u(tt,xx))
#plott(xx,tt,U1-U2)
#print(xx.shape,tt.shape,U.shape)
#plott(xx,tt,U1)
#plott(xx,tt,u_ex)
#plott(xx,tt,error)

