"""
Created on 09/01/2024
@author jdh
"""

from template import *

def reflection(Zload, Z0=50):
    return np.abs((Zload - Z0) / (Zload + Z0))


loads = np.linspace(-100, 100, 1000)

plt.figure()
plt.plot(loads, reflection(loads))
plt.show()