"""
Created on 26/01/2024
@author jdh
"""


import numpy as np
import matplotlib.pyplot as plt

data = np.load("./data/scans/right_14.npy")
double_dot = data[121, ...]
single_dot = data[201, ...]

score = np.load("./data/scores/right_14.npy")
fourier = np.load('./data/fourier_transforms/right_14.npy')

noise = np.load('/home/jdh/PycharmProjects/thesis_plots/tuning/python/data/noise/noise.npy')[9]


shape = noise.shape


plt.figure()
plt.imshow(double_dot.T, origin='lower', aspect='auto', cmap='hot')
plt.show()

# noise = np.random.normal(size=double_dot.size)

# double_dot = (double_dot - double_dot.mean()) / np.sqrt(np.var(double_dot))

dot_count, dot_bins = np.histogram(double_dot, bins=1000)
pdf = dot_count / np.sum(dot_count)

noise_count, noise_bins = np.histogram(noise, bins=1000)
pdf_noise = noise_count / np.sum(noise_count)

cdf_dot = np.cumsum(pdf)
cdf_noise = np.cumsum(pdf_noise)

#
# plt.figure()
# plt.hist(noise.flatten(), bins=100, alpha=0.7)
# plt.hist(double_dot.flatten(), bins=100, alpha=0.7)
# plt.show()


diff = np.abs(cdf_dot - cdf_noise)
max_arg = np.argmax(diff)

# plt.figure()
# plt.plot(cdf_dot, label='Dot')
# plt.plot(cdf_noise, label='Noise')
# plt.vlines(max_arg, cdf_dot[max_arg], cdf_noise[max_arg
# ], linestyles='dashdot', color='black')
# plt.legend()
#
# plt.savefig('./classifier/ks_stat.svg')
# plt.show()
# #
# plt.figure()
# plt.gca().set_aspect('equal')
# plt.scatter(noise.flatten(), double_dot.flatten())
# plt.show()


data = noise.flatten() + 1j * double_dot.flatten()

data = data * np.exp(1j * 0.8)

cov = np.cov(data.real, data.imag)
eig_values, eig_vectors = np.linalg.eig(cov)

v1 = 0.1 * np.array([
    [0, 0],
    eig_vectors[0]
])

v2 = 0.1 * np.array([
    [0, 0],
    eig_vectors[1]
])

#
# plt.figure()
# plt.gca().set_aspect('equal')
# plt.scatter(data.real, data.imag, marker='.')
# plt.scatter(data.real + 0.009, data.imag + 0.007, alpha=0.1, marker='.', color='cyan')
# plt.plot([0, 0.009], [0, 0.007], color='black', linestyle='dashed', alpha=0.4)
# plt.plot([0, 0], 0.01 * eig_vectors.T[0], color='black')
# plt.plot(*v1.T, color='black')
# plt.plot(*v2.T, color='black')
# plt.savefig('./pca.svg')
# plt.show()

#
# plt.figure()
# plt.hist(data.real.flatten(), bins=100)
# plt.savefig('./real_hist.svg')
# plt.show()
#
# plt.figure()
# plt.hist(data.imag.flatten(), bins=100)
# plt.savefig('./imag_hist.svg')
# plt.show()
#

# plt.figure(figsize=(3.5, 3.5))
# plt.imshow(data.real.reshape(shape).T, origin='lower', aspect='auto', cmap='hot', extent=(-50, 50, -50, 50)
#            )
# plt.ylabel('$\Delta V_4$ (mV)')
# plt.xlabel('$\Delta V_2$ (mv)')
# plt.savefig('./real.svg')
# plt.show()
#
# plt.figure(figsize=(3.5, 3.5))
# plt.imshow(data.imag.reshape(shape).T, origin='lower', aspect='auto', cmap='hot',
#            extent=(-50, 50, -50, 50))
# plt.ylabel('$\Delta V_4$ (mV)')
# plt.xlabel('$\Delta V_2$ (mv)')
# plt.savefig('./imag.svg'
#             )
# plt.show()

#
# # plt.figure(figsize=(3.5, 3.5))
# plt.figure()
# plt.gca().set_aspect('equal')
# plt.imshow(noise.T, origin='lower', aspect='auto', cmap='hot',
#            extent=(-50, 50, -50, 50))
# plt.ylabel('$\Delta V_4$ (mV)')
# plt.xlabel('$\Delta V_2$ (mv)')
# plt.colorbar()
# plt.savefig('./classifier/noise.svg'
#             )
# plt.show()


# # plt.figure(figsize=(3.5, 3.5))
# plt.figure()
# plt.gca().set_aspect('equal')
# plt.imshow(double_dot.T, origin='lower', aspect='auto', cmap='hot',
#            extent=(-50, 50, -50, 50))
# plt.ylabel('$\Delta V_4$ (mV)')
# plt.xlabel('$\Delta V_2$ (mv)')
# plt.colorbar()
# plt.savefig('./classifier/dot.svg'
#             )
# plt.show()

#
# plt.figure()
# plt.hist(double_dot.flatten(), bins=100)
# plt.savefig('./classifier/dd_hist.svg')
# plt.show()
#
# plt.figure()
# plt.hist(noise.flatten(), bins=100)
# plt.savefig('./classifier/imag_hist.svg')
# plt.show()


c1 = '#2D3142'
c2 = '#EF8354'

# Create the main figure and the subplots
fig, axs = plt.subplots(2, 3, gridspec_kw={'height_ratios': [3, 1]})

# Remove the empty subplot at (0, 2)
# fig.delaxes(axs[0, 2])

# Subplots in the 2x2 grid
axs[0, 0].imshow(double_dot.T, origin='lower', aspect='auto', cmap='hot',
           extent=(-50, 50, -50, 50))

axs[0, 0].set_ylabel('$\Delta V_4$ (mV)')
axs[0, 0].set_xlabel('$\Delta V_2$ (mv)')

# axs[0, 0].set_title('Subplot 1')

axs[0, 1].imshow(noise.T, origin='lower', aspect='auto', cmap='hot',
           extent=(-50, 50, -50, 50))
# axs[0, 1].set_title('Subplot 2')
axs[0, 1].set_ylabel('$\Delta V_4$ (mV)')
axs[0, 1].set_xlabel('$\Delta V_2$ (mv)')


axs[1, 0].hist(double_dot.flatten(), bins=100, color=c1)
# axs[1, 0].set_title('Subplot 3')
axs[1, 0].set_ylabel('Counts')
axs[1, 0].set_xlabel('$V_\mathrm{meas}$ (a. u.)')



axs[1, 1].hist(noise.flatten(), bins=100, color=c2)
axs[1, 1].set_ylabel('Counts')
axs[1, 1].set_xlabel('$V_\mathrm{meas}$ (a. u.)')

# axs[1, 1].set_title('Subplot 4')

# Subplot in the fifth column, taking up two rows
# axs[0, 2] = plt.subplot2grid((2, 3), (0, 2), rowspan=2)

x1 = dot_bins[:-1]
x2 = noise_bins[:-1]

y1 = cdf_dot
y2 = cdf_noise

y1_interpolated = np.interp(x2, x1, y1)
result = y2 - y1_interpolated



axs[0, 2].plot(x1, y1_interpolated, label='Dot', color=c1)
axs[0, 2].plot(x2, y2, label='Noise', color=c2)
# axs[0, 2].vlines(max_arg, cdf_dot[max_arg], cdf_noise[max_arg], linestyles='dashdot', color='black')
# axs[0, 2].set_title('Subplot 5')
axs[0, 2].set_ylabel('eCDF')
axs[0, 2].set_xlabel('$V_\mathrm{meas}$ (a. u.)')

# /
axs[1, 2].plot(x2, np.abs(result))
axs[1, 2].vlines(x2[np.argmax(result)], 0, result[np.argmax(result)], color='black', linestyle='dashdot')
axs[1, 2].set_ylabel('$\Delta$ eCDF')
axs[1, 2].set_xlabel('$V_\mathrm{meas}$ (a. u.)')

# Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.savefig('./classifier/classifier_figure.svg')
# Show the figure
plt.show()


