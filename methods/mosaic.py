import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

plt.grid(False)
# Define the dimensions
num_blocks = 5
block_size = 10
array_size = num_blocks * block_size

# Create the base array with the block values
block_values = np.arange(num_blocks**2).reshape(num_blocks, num_blocks)

# Repeat the block values to fill the array
array = np.kron(block_values, np.ones((block_size, block_size), dtype=int))

x = np.linspace(block_size // 2, array_size - block_size // 2, num_blocks)
y = np.linspace(block_size // 2, array_size - block_size // 2, num_blocks)

X, Y = np.meshgrid(x, y)

long_ramp = np.linspace(0, 1, 900)
long_ramp[-1] = 0
short_ramp = np.linspace(0, 1, 100)
ramps = np.concatenate([short_ramp] * 9)



plt.subplot(2, 2, 1)
plt.imshow(array.T, origin='lower', cmap='Blues')
plt.scatter(X, Y, color='brown', s=2)
# plt.savefig('./test.pdf')


plt.subplot(2, 2, 2)
plt.plot(long_ramp + 1)
plt.plot(ramps - 1)


# subfigure
plt.subplot(2, 2, 3)
u = np.linspace(0, 1, 81).reshape(9, 9)
plt.imshow(u.T, origin='lower', cmap='Blues')
plt.colorbar()




