import numpy as np

# arr = np.array([[1.9192,2.8921,3.921,4.911],
#                 [5.82,6.872,7.82,8.02]])

# print(arr.shape)
# print(arr.size)
# print(arr.ndim)
# print(arr.dtype)
# print(arr.astype(int))


# math

arr = np.array([10,20,30,40,50,60,70,80,90,100])
# print(arr+5)

# print(np.sum(arr))
# print(np.mean(arr))
# print(np.max(arr))
# print(np.min(arr))
# print(np.std(arr))
# print(np.var(arr))


# x = np.arange(0,201,5) #start, stop , step 
# print(x)

# arr2 = np.linspace(0,100,5)
# print(arr2)

# arr3 = np.logspace(2,3,5)
# print(arr3)

# print(arr[3:])
# print(arr[4:6])
# print(arr[3:])
# print(arr[:-1])
# print(arr[::-1])
# print(arr[::2])

# reshapint = arr.reshape(3,3)
# print(reshapint)


# print(reshapint.flatten())
# print(reshapint.ravel())

# arr = np.array([10,20,30,40,50,60,70,80,90])
# print(arr)
# new_arr = np.insert(arr, 4, 2000)
# print(new_arr)

# arr_2d = np.array([[1,2,3],
#                    [4,5,6]])

# new_arr = np.insert(arr_2d, 1 , [5,6,7],axis=None)
# print(new_arr)

# new_arr2 = np.append(arr_2d,[7,8,9])
# print(new_arr2)

# arr = np.array([1,2,3,4,5])
# arr2 = np.array([6,7,8,9,10])
# new_array = np.concatenate((arr , arr2))
# print(new_array)

# arr = np.array([1,2,3,4,5,6,7])
# remove = np.delete(arr,4)
# print(remove)

# arr2d = np.array([[1,2,3],
#                   [4,5,6]])
# new_array = np.delete(arr2d, 1 , axis=0)
# print(new_array)

# arr = np.array([1,2,3])
# arr2 = np.array([4,5,6])

# print(np.vstack((arr,arr2)))
# print(np.hstack((arr,arr2)))


# arr = np.array([10,20,30,40,50,60])
# print(np.split(arr,2))
# print(np.hsplit(arr,2))
# print(np.vsplit())


# prices = np.array([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900])
# discount = 15
# finalprices = prices - (prices * 10 / 100)
# print(finalprices)

# arr = np.array([10,20,30])
# result = arr * 2
# print(result)

# x = np.array([[10,20,30],[40,50,60]])
# y = np.array([[70,80,90],[100,110,120]])
# result = x + y 
# print(result)


# arr = np.array([1,2,3,np.nan,np.nan,6,7,np.nan])
# x = np.isnan(arr)
# print(x)

# x =  np.nan_to_num(arr,nan=12)
# print(x)

arr = np.array([1,2,3,np.inf,np.inf,6,7,-np.inf])
x = np.isinf(arr)
print(x)

y = np.nan_to_num(arr,posinf=43,neginf=10)
print(y)