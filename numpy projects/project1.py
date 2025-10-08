import numpy as np
import time


# print("=" * 60)
# print("NUMPY MASTERY PROJECT - DATA ANALYSIS SUITE")
# print("=" * 60)

# print("\n[PART 1] Sales Data Analysis")
# print("-" * 60)

# np.random.seed(42)
# sales_data = np.random.randint(100, 1000, size=(15,5,3))
# print(f"sales data shape: {sales_data.shape}")
# print(f"(Months, Products, Regions)\n")

# total_per_product = np.sum(sales_data, axis=(0,2))
# print(f"total sales per product: {total_per_product}")

# best_month_per_product_region = np.argmax(sales_data, axis=0)
# print(F"\nBest month index for each product-region:\n{best_month_per_product_region}")

# monthly_totals = np.sum(sales_data,axis=(1,2))
# growth_rate = np.diff(monthly_totals) / monthly_totals[:-1] * 100
# print(f"\nMonth-over-month growth rate (%):\n{np.round(growth_rate, 2)}")

# q1_months = [0, 1, 2]
# q1_sales = sales_data[[0, 1, 2],:,:]
# print(f"\nQ1 Total Sales: ${np.sum(q1_sales):,}")




print("\n\n[PART 2] Statistical Analysis with Broadcasting")
print("-" * 60)

np.random.seed(43)
grades = np.random.randint(50,100,size=(100,5))

min_grades = np.min(grades,axis=0)
max_grades = np.max(grades,axis=0)
normalized = (grades - min_grades) / (max_grades - min_grades)

# print(f"Original grade sample:\n{grades[:3]}")
# print(f"\nNormalized grade sample:\n{normalized[:3]}")


# mean = np.mean(grades,axis=0)
# std = np.std(grades,axis=0)
# z_scores = (grades- mean) / std
# print(f"\nMean grades per subject: {np.round(mean,2)}")
# print(f"std dev per subject: {np.round(std,2)}")

weights = np.array([0.15, 0.20, 0.25, 0.20, 0.20])
weighted_avg = np.sum(grades * weights,axis=1)
print(f"\nTop 5 students(weighted avg):\n{np.sort(weighted_avg)[-5:]}")

failed_students = np.any(grades < 60 , axis=1)
print(f"\nnumber of students who failed at least onk subject: {np.sum(failed_students)}")