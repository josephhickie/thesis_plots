"""
Created on 14/06/2023
@author jdh
"""

import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots
import matplotlib

plt.style.use(['science', 'ieee', 'bright', 'grid', 'no-latex'])
font = {'size': 10}
matplotlib.rc('font', **font)
# plt.style.use('/home/jdh/PycharmProjects/thesis_plots/src/settings.txt')

def exponential(t, a, b, T1):
    return a * np.exp(-t / T1) + b

wait_times = np.arange(16, 40000, 64)

wait_times = wait_times / 1000

with open('./t1.npy', 'rb') as f:
    I = np.load(f)
    Q = np.load(f)

R = np.sqrt(I**2 + Q**2) * 1000
r = curve_fit(exponential, wait_times, R, p0=[1, 0, 10000])

params = r[0]
T1 = params[2]

plt.figure(figsize=(6, 4))
plt.vlines(T1, 0.15, exponential(T1, *r[0]), 'black', 'dashed', alpha=0.5)
plt.hlines(exponential(T1, *r[0]), 0, T1, 'black', 'dashed', alpha=0.5, label='$T_1$')

plt.plot(wait_times, R, alpha=0.8, color='grey', label='Data')
plt.plot(wait_times, exponential(wait_times, *r[0]), color='red', label='Exponential fit')
plt.legend()
plt.xlabel("Wait time / us")
plt.ylabel("R / V")


plt.xlim(0, 25)
plt.ylim(0.15, 0.4)
plt.show()

plt.savefig('./T1_fit_2.pdf')
