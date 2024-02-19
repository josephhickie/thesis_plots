import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

with open('./couplinG_vals.npy', 'rb') as f:
    dataset = np.load(f)

# Define parameters
sample_sizes = np.arange(1, len(dataset)+1)
num_samples_per_size = 30  # Number of times to sample each sub sample size

# Initialize lists to store mean and standard deviation
means = []
std_devs = []

# Iterate over different sub sample sizes
for size in sample_sizes:
    sample_means = []
    sample_std_devs = []
    # Sample multiple times for each sub sample size
    for _ in range(num_samples_per_size):
        sample = np.random.choice(dataset, size=size, replace=False)
        sample_means.append(np.mean(sample))
        # sample_std_devs.append(np.std(sample))
    # Calculate mean and standard deviation of sample means
    means.append(np.mean(sample_means))
    std_devs.append(np.std(sample_means))

means = np.array(means)
std_devs = np.array(std_devs)


fig, axs = plt.subplots(1, 2)

# Plotting
# axs[0].figure(figsize=(10, 6))
# plt.errorbar(sample_sizes, means, yerr=std_devs, fmt='o', linestyle='-')
axs[0].plot(sample_sizes, means)
axs[0].fill_between(sample_sizes, means - std_devs, means + std_devs, alpha=0.2)

axs[0].set_xlabel('Sample Size')
axs[0].set_ylabel('Mean')
axs[0].set_title('Convergence of coupling value mean with sample size')
axs[0].grid(True)

axs[1].hist(dataset.flatten(), bins=100)
plt.show()