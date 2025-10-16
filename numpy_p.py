import numpy as np
import time

# ============================================
# NumPy Expert Project: Data Analysis Suite
# ============================================
# This project will help you master NumPy through practical applications

print("=" * 60)
print("NUMPY MASTERY PROJECT - DATA ANALYSIS SUITE")
print("=" * 60)

# ============================================
# PART 1: Sales Data Analysis (Advanced Array Operations)
# ============================================
print("\n[PART 1] Sales Data Analysis")
print("-" * 60)

# Create synthetic sales data for a company
# 12 months, 5 products, 3 regions
np.random.seed(42)
sales_data = np.random.randint(100, 1000, size=(12, 5, 3))

print(f"Sales Data Shape: {sales_data.shape}")
print(f"(Months, Products, Regions)\n")

# Task 1.1: Calculate total sales per product across all months and regions
total_per_product = np.sum(sales_data, axis=(0, 2))
print(f"Total sales per product: {total_per_product}")

# Task 1.2: Find best performing month for each product in each region
best_month_per_product_region = np.argmax(sales_data, axis=0)
print(f"\nBest month index for each product-region:\n{best_month_per_product_region}")

# Task 1.3: Calculate percentage growth month-over-month
monthly_totals = np.sum(sales_data, axis=(1, 2))
growth_rate = np.diff(monthly_totals) / monthly_totals[:-1] * 100
print(f"\nMonth-over-month growth rate (%):\n{np.round(growth_rate, 2)}")

# Task 1.4: Use fancy indexing to get Q1 sales (Jan, Feb, Mar)
q1_months = [0, 1, 2]
q1_sales = sales_data[q1_months, :, :]
print(f"\nQ1 Total Sales: ${np.sum(q1_sales):,}")


# ============================================
# PART 2: Statistical Analysis & Broadcasting
# ============================================
print("\n\n[PART 2] Statistical Analysis with Broadcasting")
print("-" * 60)

# Create student grades dataset: 100 students, 5 subjects
grades = np.random.randint(50, 100, size=(100, 5))

# Task 2.1: Normalize grades using broadcasting (0-100 to 0-1 scale)
min_grades = np.min(grades, axis=0)
max_grades = np.max(grades, axis=0)
normalized = (grades - min_grades) / (max_grades - min_grades)
print(f"Original grade sample:\n{grades[:3]}")
print(f"\nNormalized grade sample:\n{normalized[:3]}")

# Task 2.2: Calculate z-scores for each subject
mean = np.mean(grades, axis=0)
std = np.std(grades, axis=0)
z_scores = (grades - mean) / std
print(f"\nMean grades per subject: {np.round(mean, 2)}")
print(f"Std dev per subject: {np.round(std, 2)}")

# Task 2.3: Apply weights to subjects and calculate weighted average
weights = np.array([0.15, 0.20, 0.25, 0.20, 0.20])  # Sum = 1.0
weighted_avg = np.sum(grades * weights, axis=1)
print(f"\nTop 5 students (weighted avg):\n{np.sort(weighted_avg)[-5:]}")

# Task 2.4: Boolean indexing - find students who failed any subject (< 60)
failed_students = np.any(grades < 60, axis=1)
print(f"\nNumber of students who failed at least one subject: {np.sum(failed_students)}")


# ============================================
# PART 3: Image Processing Simulation (Advanced Indexing)
# ============================================
print("\n\n[PART 3] Image Processing Simulation")
print("-" * 60)

# Create a synthetic "image" (grayscale 100x100)
image = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

# Task 3.1: Extract and process image patches
patch_size = 10
patches = []
for i in range(0, 100, patch_size):
    for j in range(0, 100, patch_size):
        patch = image[i:i+patch_size, j:j+patch_size]
        patches.append(patch)
patches_array = np.array(patches)
print(f"Extracted {len(patches)} patches of size {patch_size}x{patch_size}")

# Task 3.2: Calculate mean intensity per patch
mean_intensities = np.mean(patches_array, axis=(1, 2))
print(f"Average intensity across all patches: {np.round(np.mean(mean_intensities), 2)}")

# Task 3.3: Apply threshold (binary image)
threshold = 128
binary_image = np.where(image > threshold, 255, 0)
print(f"Percentage of bright pixels: {np.sum(binary_image == 255) / binary_image.size * 100:.2f}%")

# Task 3.4: Create a border around the image
bordered_image = np.pad(image, pad_width=5, mode='constant', constant_values=255)
print(f"Original shape: {image.shape}, Bordered shape: {bordered_image.shape}")


# ============================================
# PART 4: Matrix Operations & Linear Algebra
# ============================================
print("\n\n[PART 4] Matrix Operations & Linear Algebra")
print("-" * 60)

# Task 4.1: Create a correlation matrix
data_points = np.random.randn(1000, 4)  # 1000 samples, 4 features
correlation_matrix = np.corrcoef(data_points.T)
print(f"Correlation Matrix:\n{np.round(correlation_matrix, 3)}")

# Task 4.2: Solve a system of linear equations (Ax = b)
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)
print(f"\nSolving 3x + y = 9 and x + 2y = 8:")
print(f"Solution: x = {x[0]}, y = {x[1]}")

# Task 4.3: Calculate eigenvalues and eigenvectors
matrix = np.array([[4, -2], [1, 1]])
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Task 4.4: Matrix multiplication and transpose operations
X = np.random.randn(50, 3)  # 50 samples, 3 features
weights = np.array([0.5, -0.3, 0.8])
predictions = X @ weights  # Matrix multiplication
print(f"\nPredictions shape: {predictions.shape}")


# ============================================
# PART 5: Time Series Analysis (Advanced)
# ============================================
print("\n\n[PART 5] Time Series Analysis")
print("-" * 60)

# Generate synthetic time series data (1 year of daily data)
days = 365
time = np.arange(days)
trend = 0.1 * time
seasonality = 10 * np.sin(2 * np.pi * time / 365)
noise = np.random.randn(days) * 2
time_series = 100 + trend + seasonality + noise

# Task 5.1: Calculate moving average
window = 7
moving_avg = np.convolve(time_series, np.ones(window)/window, mode='valid')
print(f"Original time series length: {len(time_series)}")
print(f"Moving average length (window={window}): {len(moving_avg)}")

# Task 5.2: Find peaks (values above 90th percentile)
threshold_90 = np.percentile(time_series, 90)
peaks = time_series > threshold_90
print(f"\nNumber of peak days (>90th percentile): {np.sum(peaks)}")

# Task 5.3: Calculate autocorrelation at lag 7
lag = 7
autocorr = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
print(f"Autocorrelation at lag {lag}: {autocorr:.3f}")

# Task 5.4: Detect anomalies using z-score
z_scores_ts = np.abs((time_series - np.mean(time_series)) / np.std(time_series))
anomalies = z_scores_ts > 3
print(f"Number of anomalies (|z| > 3): {np.sum(anomalies)}")


# ============================================
# PART 6: Performance Optimization Challenge
# ============================================
print("\n\n[PART 6] Performance Optimization Challenge")
print("-" * 60)

# Challenge: Compare vectorized vs loop-based operations
large_array = np.random.randn(1000000)

# Method 1: Vectorized (NumPy way)
start = time.time()
result_vectorized = np.sqrt(np.abs(large_array)) + np.power(large_array, 2)
time_vectorized = time.time() - start

# Method 2: Loop (Python way - DON'T DO THIS!)
start = time.time()
result_loop = np.zeros_like(large_array)
for i in range(len(large_array)):
    result_loop[i] = np.sqrt(abs(large_array[i])) + large_array[i] ** 2
time_loop = time.time() - start

print(f"Vectorized time: {time_vectorized:.4f} seconds")
print(f"Loop time: {time_loop:.4f} seconds")
print(f"Speedup: {time_loop / time_vectorized:.1f}x faster with vectorization!")
