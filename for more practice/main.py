import numpy as np 


result = np.array([
 [80, 78, 87, 84, 83, 83, 76, 86, 71, 77, 75, 84],
 [77, 80, 81, 74, 77, 72, 76, 81, 79, 77, 84, 82],
 [81, 89, 73, 79, 82, 85, 73, 78, 88, 71, 85, 70],
 [74, 87, 76, 72, 89, 73, 87, 81, 80, 70, 82, 70],
 [84, 87, 74, 88, 76, 86, 72, 89, 72, 84, 86, 80],
 [76, 86, 72, 71, 77, 71, 70, 86, 70, 73, 75, 74],
 [77, 75, 75, 73, 88, 89, 81, 75, 73, 85, 72, 86],
 [70, 82, 76, 76, 87, 82, 71, 80, 80, 73, 83, 86],
 [78, 73, 80, 86, 86, 71, 74, 84, 77, 70, 72, 74],
 [88, 86, 88, 73, 85, 84, 86, 78, 72, 87, 74, 84],])


avg_per_student = result.mean(axis=1)
avg_per_subjects = result.mean(axis=0)
higest_mark = result.max()
lowes_marks = result.min()


print(np.round(avg_per_student,2))
print(avg_per_subjects)
print(higest_mark)
print(lowes_marks)
