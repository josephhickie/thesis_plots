"""
Created on 12/10/2023
@author jdh
"""

from template import *

x = np.linspace(0, 60, 10000)

def lorentz(x, gamma, x0=0):
    return (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2)



gamma = 1
x0s = np.arange(9, 60, 15)

sum = np.zeros_like(x)
for x0 in x0s:
    sum += lorentz(x, gamma, x0)



gradient = np.gradient(sum)
grad_threshold = 1e-4

sensitive = np.abs(gradient) > grad_threshold


plt.figure()
plt.fill_between(x, np.min(sum), np.max(sum), where=sensitive, alpha=0.2, label='Sensitive')
plt.fill_between(x, np.min(sum), np.max(sum), where=np.logical_not(sensitive), alpha=0.2, label='Not sensitive')
plt.plot(x, sum)
plt.xlabel('Electrostatic potential (arb. units)')
plt.ylabel('Sensor output (arb. units)')
plt.legend()
plt.savefig('./sensitivity.pdf')
plt.show()

