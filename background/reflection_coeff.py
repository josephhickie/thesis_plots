"""
Created on 18/01/2024
@author jdh
"""
import matplotlib.pyplot as plt

from template import *

def gamma(z, z0=50):
    return np.abs((z - z0) / (z + z0))

zs = np.linspace(0, 50e3, 2000)
z0 = 50

plt.figure()
plt.plot(zs / z0, gamma(zs))
plt.xlabel('$Z_\mathrm{load} / Z_0$')
plt.ylabel('$|\Gamma|$')
plt.vlines(26e3/50, 0, 1, color='black', alpha=0.4)
plt.savefig('./large_z.pdf')
plt.show()


plt.figure()
plt.plot(zs / z0, np.abs(np.gradient(gamma(zs))))
plt.xlabel('$Z_\mathrm{load} / Z_0$')
plt.ylabel('$|\Gamma|$')
plt.vlines(26e3/50, 0, 1, color='black', alpha=0.4)
plt.savefig('./large_z_grad.pdf')
plt.show()


zs_small = np.linspace(0, 10 * z0, 1000)

plt.figure()
plt.plot(zs_small/ z0, gamma(zs_small))
plt.xlabel('$Z_\mathrm{load} / Z_0$')
plt.ylabel('$|\Gamma|$')
plt.savefig('./small_z.pdf')
plt.show()

