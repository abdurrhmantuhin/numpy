import numpy as np

# arr = np.array([[1.9192,2.8921,3.921,4.911],
#                 [5.82,6.872,7.82,8.02]])

# print(arr.shape)
# print(arr.size)
# print(arr.ndim)
# print(arr.dtype)
# print(arr.astype(int))


# math

# arr = np.array([10,20,30,40,50,60,70,80,90,100])
# print(arr+5)

# print(np.sum(arr))
# print(np.mean(arr))
# print(np.max(arr))
# print(np.min(arr))
# print(np.std(arr))
# print(np.var(arr))


x = np.arange(0,100,5) #start, stop , step 
print(x)

arr2 = np.linspace(0,100,5)
print(arr2)

arr3 = np.logspace(2,3,5)
print(arr3)

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

# arr = np.array([1,2,3,np.inf,np.inf,6,7,-np.inf])
# x = np.isinf(arr)
# print(x)

# y = np.nan_to_num(arr,posinf=43,neginf=10)
# print(y)






















# # Create a 3D array (3 pages, 4 rows, 5 columns)
# arr_3d = np.array([
#     # Page 0
#     [[1,  2,  3,  4,  5],
#      [6,  7,  8,  9,  10],
#      [11, 12, 13, 14, 15],
#      [16, 17, 18, 19, 20]],
    
#     # Page 1
#     [[21, 22, 23, 24, 25],
#      [26, 27, 28, 29, 30],
#      [31, 32, 33, 34, 35],
#      [36, 37, 38, 39, 40]],
    
#     # Page 2
#     [[41, 42, 43, 44, 45],
#      [46, 47, 48, 49, 50],
#      [51, 52, 53, 54, 55],
#      [56, 57, 58, 59, 60]]
# ])

# print("Shape:", arr_3d.shape)  


# # 1. Access a single element
# element = arr_3d[1, 2, 3]  # Page 1, Row 2, Column 3
# print("Single element:", element)  # 34

# # 2. Access an entire page (2D array)
# page_0 = arr_3d[]  # All of page 0
# print("\nPage 0:\n", page_0)

# # 3. Access a specific row from a specific page
# row = arr_3d[1, 2]  # Page 1, Row 2
# print("\nPage 1, Row 2:", row)  # [31, 32, 33, 34, 35]

# # 4. Access a specific column from all pages
# column = arr_3d[:,0,3]  # All pages, all rows, column 2
# print("\nColumn 2 from all pages:\n", column)

# # 5. Access using slicing
# slice_result = arr_3d[0:2, 1:3, 2:4]  # Pages 0-1, Rows 1-2, Columns 2-3
# print("\nSliced array:\n", slice_result)

# # 6. Negative indexing works too
# last_element = arr_3d[-1, -1,-1]  # Last page, last row, last column
# print("\nLast element:", last_element)  # 60








# # Create a 4D array: (2, 2, 3, 4)
# # 2 bookshelves, 2 books each, 3 pages each, 4 lines each
# arr_4d = np.array([[[[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]],[[13, 14, 15, 16],[17, 18, 19, 20],[21, 22, 23, 24]]],
# [[[25, 26, 27, 28],[29, 30, 31, 32],[33, 34, 35, 36]],[[37, 38, 39, 40],[41, 42, 43, 44],[45, 46, 47, 48]]]])

# # print("Shape:", arr_4d.shape)  # (2, 2, 3, 4)
# # print("This means: 2 bookshelves, 2 books, 3 pages, 4 lines\n")

# # bookshelf = arr_4d[0]
# # print(bookshelf)

# arr = arr_4d[1,1,2,3]

# print('the output is:',arr)