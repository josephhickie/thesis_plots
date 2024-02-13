"""
Created on 29/01/2024
@author jdh
"""


import pickle
# from template import *
import matplotlib.pyplot as plt

with open('./data_for_nonlinear.p', 'rb') as f:
    data = pickle.load(f)

x = data.get('x')
y = data.get('y')
z = data.get('z')

plt.figure()
plt.imshow(z.T, origin='lower', aspect='auto', extent=(x[0, 0], x[-1, -1], y[0, 0], y[-1, -1]), cmap='hot')
plt.xlabel('$V_P$ (mV)')
plt.ylabel('$V_\mathrm{gate}$ (mV)')
plt.colorbar()
plt.savefig('./non-linear.svg')
plt.show()
