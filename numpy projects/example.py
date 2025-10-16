import numpy as np
import time

## creating arrays
# arr_1d = np.array(
#     [1,2,3,4,5]
#     )

# arr_2d = np.array([
#     [1,2,3,4,5],
#     [6,7,8,9,10]
# ])

# arr_3d = np.array([
#     [1,2,3,4,5],
#     [6,7,8,9,10],
#     [11,12,13,14,15]
#     ])

# print(F"\n1D Array: \n{arr_1d}\n")
# print(F"2D Array: \n{arr_2d}\n")
# print(F"3D Array: \n{arr_3d}\n")





## eliment wise multiplication
# py_array = [1,2,3,4,5]
# np_array = np.array([1,2,3,4,5])

# result_py = py_array * 2
# result_np = np_array * 2

# print(f"python list multiplication by 2: {result_py}")
# print(f"numpy array multiplication by 2: {result_np}")




## how to work numpy faster than python list
# strart = time.time()
# py_list = [i * 2 for i in range(100000000)]
# print(f"list comprehension time: {time.time() - strart}")

# start2 = time.time()
# np_array2 = np.arange(100000000) * 2
# print(f"numpy array operation time: {time.time() - start2}")





## creating arrau from scratch
# zeros = np.zeros((3,2,5))   # only zeros(0) [1st:dimension, 2nd:rows, 3rd:columns]
# print(zeros)

# ones = np.ones((3,2,5))  # only one(1) [1st:dimension, 2nd:rows, 3rd:columns]
# print(ones)

# full = np.full((3,2,5),20)  # you can fill any number you want [1st:dimension, 2nd:rows, 3rd:columns, value]
# print(full) 

# random = np.random.random((3,2,5)) # random values between 0 and 1 [1st:dimension, 2nd:rows, 3rd:columns]
# print(random)

# sequence = np.arange(0,500,5) 
# print(sequence)  





## vectors, matrices and tensor
# vector = np.array([1,2,3])  # 1D array
# print(f"vector: {vector}\n")

# matrix = np.array([         # 2D array
#     [1,2,3],
#     [4,5,6],
# ])
# print(f"matrix: \n{matrix}\n")


# tensor = np.array([         # multi dimensional array                     
#     [[1,2,3,],[4,5,6],],
#     [[7,8,9],[10,11,12],],
#     [[13,14,15],[16,17,18],],
# ])
# print(f"tensor: \n{tensor}\n")





## array properties
# array = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9],
# ])
# print(f"shape:{array.shape}")  # (rows, columns)
# print(F"dimansion: {array.ndim}")  # number of dimensions
# print(f"size: {array.size}")  # total number of elements
# print(f"data type: {array.dtype}")  # data type of elements 





## array  reshaping
# array_1 = np.arange(12)
# print(array_1)

# reshaped = array_1.reshape(3,4)
# print(reshaped)

# flattened = reshaped.flatten()      # flatten returns copy of original array
# print(flattened)

# raveled = reshaped.ravel()          # ravel retirns view of original array
# print(raveled)

# transposed = reshaped.T            # transpose returns view of original array
# print(transposed)