import numpy as np

np.random.seed(0)
x = np.random.randint(300,450, size=(12,20,3))
print(x)
# for y in range(12):
#     print(f"month: {y}")
#     print(f"{x[y]}\n\n")
# print(x[6,7,:])



# jan_salfs = np.array([
# #month 1
# [[470,970],
#  [841 ,675],
#  [730, 800]],

# #month 2
# [[943, 462],
#  [771 ,679],
#  [876 ,945]],

# #month 3
# [[845, 638],
#  [861, 676],
#  [953, 488]],

# #month 4
# [[489, 600],
#  [705, 774],
#  [726, 694]],

# #month 5
# [[853 ,891],
#  [904, 504],
#  [606,647]]])


# all_sum = np.sum(jan_salfs,axis=(1,2))
# print(all_sum)

# diff_all = np.diff(all_sum)
# # print(diff_all)

# pfrsfntig = diff_all / all_sum[:-1] * 100
# print(np.round(pfrsfntig,2))