"""
Created on 28/06/2023
@author jdh
"""

import numpy as np
import matplotlib as mpl
import sys
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import scienceplots
import matplotlib


plt.style.use(['science', 'ieee', 'bright', 'grid', 'no-latex'])
font = {'size': 10}
matplotlib.rc('font', **font)


# sys.path.remove( '/home/jdh/PycharmProjects/qgor')
sys.path.append('/home/jdh/PycharmProjects/qgor_qm')