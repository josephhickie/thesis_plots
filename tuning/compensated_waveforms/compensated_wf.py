"""
Created on 09/01/2024
@author jdh
"""


from template import *

sample_rate = 1e7
time = 10e-6

samples = int(time * sample_rate)


resolution = 10
samples_per_point = int(samples / resolution)

t = np.linspace(0, time, resolution)
x = np.ones((resolution, samples_per_point))

times = np.linspace(0, time, samples)

step = (t[:, np.newaxis] * x).flatten()

plt.figure()
plt.plot(times, step)
plt.show()






# fft
from numpy.fft import fft, ifft

X = fft(step)
N = len(X)
n = np.arange(N)
T = N / sample_rate
freq = n / T

plt.plot()
plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)
plt.show()

plt.subplot(122)
plt.plot(times, ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
