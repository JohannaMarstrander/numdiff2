# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:29:52 2020

@author: johan
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def f(x, t, params):
    S, I, R  = x
    beta, gamma = params
    return [-beta*S*I, beta*S*I - gamma*I, gamma*I]

beta = 1.0
gamma = 1.0

S0 = 0.999
I0 = 0.001
R0 = 0.0

params = [beta, gamma]
init = [S0,I0,R0]

# Make time array for solution
tStop = 100.
tInc = 0.05
t = np.arange(0., tStop, tInc)

psoln = odeint(f, init, t, args=(params,))

# Plot results
fig = plt.figure(1, figsize=(8,8))

# Plot S as a function of time
ax1 = fig.add_subplot(311)
ax1.plot(t, psoln[:,1], color = "green")
ax1.set_xlabel('time')
ax1.set_ylabel('S,I,R')

# Plot I as a function of time
ax2 = fig.add_subplot(311)
ax2.plot(t, psoln[:,0], color = "red")


# Plot R as a function of time
ax3 = fig.add_subplot(311)
ax3.plot(t, psoln[:,2], color = "blue")


# Plot omega vs theta
ax4 = fig.add_subplot(312)
ax4.plot(t, psoln[:,1]+psoln[:,2] + psoln[:,0], '.', ms=1)

plt.tight_layout()
plt.show()
