import cupy as cp
import numpy as np

# Define a simple function to be executed on the GPU
@cp.fuse()
def my_kernel(x, zeros):
    row = 0
    for i in range(10):
        zeros[row, 0:x[i]] = 1
        row = row + 1
    return zeros

# Generate some random data on CPU
N = 10

x_cpu = np.random.randint(10,size=N)

zerosarr_cpu = np.zeros((10,10)) 

# Transfer data to GPU
#x_gpu = cp.asarray(x_cpu)
#zerosarr_gpu = cp.asarray(zerosarr)

# Perform the computation on the GPU
result_gpu = my_kernel(x_cpu, zerosarr_cpu)

# Transfer the result back to CPU if needed
result_cpu = cp.asnumpy(result_gpu)

# Print the result
print(result_cpu)
