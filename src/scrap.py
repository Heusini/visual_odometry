import numpy as np

arr1 = np.random.rand(10)
arr2 = np.random.rand(10)

arr1 = np.atleast_1d(arr1)
arr2 = np.atleast_1d(arr2)

print(arr1)
print(arr2)





# Calculate the absolute differences and check if within tolerance
diffs = np.abs(arr1[:, None] - arr2)

print(diffs)