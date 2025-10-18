import numpy as np

# np.random.seed(1)
# result = np.random.randint(30,99,(20,10))

# student_avg = np.mean(result,axis=1)
# fails = np.any(result < 33, axis=1)
# a_plus_students = np.sum(np.sum(result >= 80, axis=1) >= 4)
# pass_student = np.sum(~fails)


# print("Student Result Dashboard")
# print("-" * 30)
# for x in range(20):
#     print(f"Student {x}: {student_avg[x]} marks")
# print(f"total pass student: {pass_student}")
# print(f"total failed student: {np.sum(fails)}")
# print(f"A+ student (students with at least 4 subjects >= 80 marks): {a_plus_students}")


# np.random.seed(2)
# temperature = np.random.randint(25, 40, (4, 7))

# avg_temp = np.mean(temperature, axis=1)
# highest = np.max(temperature, axis=1)
# lowest = np.min(temperature, axis=1)
# hot_day = np.where(temperature >= 38)

# print("\n TEMPERATURE DASHBOARD")
# print("-" * 40)
# print(f"weekly average: {np.round(avg_temp,2)}")
# print(f"weekly max: {highest}")
# print(f"weekly min: {lowest}")
# print(f"Day's ≥38°C: {len(hot_day[0])}")
# for week, day in zip (hot_day[0],hot_day[1]):
#     print(f"week {week+1} Day {day+1} = {temperature[week,day]}°C ")