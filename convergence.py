import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Generate your dataset
# Replace this with your actual dataset
dataset = np.random.randn(1000)

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

# Plotting
plt.figure(figsize=(10, 6))
# plt.errorbar(sample_sizes, means, yerr=std_devs, fmt='o', linestyle='-')
plt.plot(sample_sizes, means)
plt.fill_between(sample_sizes, means - std_devs, means + std_devs, alpha=0.2)

plt.xlabel('Sample Size')
plt.ylabel('Mean')
plt.title('Convergence of Sample Mean with Standard Deviation')
plt.grid(True)
plt.show()