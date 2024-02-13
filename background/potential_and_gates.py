"""
Created on 09/02/2024
@author jdh
"""

import matplotlib.pyplot as plt
import numpy as np



x = np.linspace(0, 5, 1000)


x2 = x[300:600]



x0 = 2.5


def squared(x, x0):
    return (x-x0)**2

def lorentz(x, gamma, x0=0):
    return (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)


inv = 5 * lorentz(x, 1, x0)
sq = squared(x, x0)

plt.figure(figsize=(4, 1))
# plt.plot(x, sq)
# plt.plot(x, inv)
plt.plot(x, inv + sq)
plt.savefig('./potential.svg')
plt.show()

