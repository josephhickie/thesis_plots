"""
Created on 28/06/2023
@author jdh

simultaneous readout data but this is not actually that important.

TODO: make the axis equal

this is a plot for showing that the second axis of pca is just the
gaussian noise and we can therefore use it to extract the noise for
the fitting. This way we do not have to fit the noise parameter.

"""

from template import *

# import sys
from helper_functions import load_np

sys.path.append('/home/jdh/PycharmProjects/readout_optimisation_cph')

from readout_opt.fitting import pca, normalise
from readout_opt import gaussian


idx = 0

data_dictionary = load_np('./2023-06-28 12:47:02.623173.npz')

I = data_dictionary.get('I')[idx]
Q = data_dictionary.get('Q')[idx]

data, noise = pca(I, Q).T

rescale = np.max(data) - np.min(data)

noise = (noise - np.min(noise)) / rescale
data = (data - np.min(data)) / rescale



def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=1, marker=',')
    ax.set_xlabel('PCA$_0$ / V')
    ax.set_ylabel('PCA$_1$ / V')

    # ax.set_aspect('equal', adjustable='box')

    ax_histx.hist(x, bins=80, density=True)
    ax_histx.set_ylabel('Density')

    std = np.std(y)
    mean = np.mean(y)
    vals = np.linspace(*ax_histy.get_ylim(), 500)
    gauss = gaussian(vals, mean, std)

    ax_histy.hist(y, bins=80, orientation='horizontal', density=True)
    ax_histy.set_xlabel('Density')
    ax_histy.plot(gauss, vals, 'k--', alpha=0.6)

fig, axs = plt.subplots(2, 2, figsize=(5, 3))

axs[0, 0].scatter(I * 1e3, Q * 1e3, s=1, marker=',')
axs[0, 0].set_xlabel('I / V')
axs[0, 0].set_ylabel('Q / V')

axs[0, 1].scatter(data, noise, s=1, marker=',')
axs[0, 1].set_xlabel('PCA$_0$ / V')
axs[0, 1].set_ylabel('PCA$_1$ / V')

axs[1, 0].hist(data, bins=80)
axs[1, 0].set_xlabel('PCA$_0$ / V')
axs[1, 0].set_ylabel('Count')

axs[1, 1].hist(noise, bins=80)
axs[1, 1].set_xlabel('PCA$_1$ / V')
axs[1, 1].set_ylabel('Count')


plt.tight_layout()

plt.savefig('./pca_gaussian.pdf')
plt.show()

# Start with a square Figure.
fig = plt.figure(figsize=(5, 4))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.1, hspace=0.1)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

# Draw the scatter plot and marginals.
scatter_hist(data, noise, ax, ax_histx, ax_histy)
plt.tight_layout()
plt.savefig('./scatter_with_hist_pca_readout.pdf')