import numpy as np

# Create 3D array with random integers between 450 and 1000
# Shape: (5, 3, 2) - 5 layers, 3 rows, 2 columns
array = np.random.randint(450, 1000, size=(5, 3, 2))


for i in range(3,5):
    print(f"\nLayer {i}:")
    print(array[i])

