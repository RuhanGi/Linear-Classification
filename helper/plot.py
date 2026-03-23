import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Error: No input CSV filename provided.")
    sys.exit(1)

input_filename = sys.argv[1]
base_name, _ = os.path.splitext(input_filename)
output_filename = f"{base_name}.png"

df = pd.read_csv(input_filename)

# 1. SPLIT DATA using your 0.0 flag
# (We use < 1e-6 instead of == 0 to avoid floating point precision bugs)
df_train = df[df['stds'] < 1e-6].copy()
df_test = df[df['stds'] >= 1e-6].copy()

# 2. Sort TEST data by the X-axis for a clean line plot
df_test = df_test.sort_values(by='true')

# 3. Calculate metrics for the TEST set
mse_test = np.mean((df_test['means'] - df_test['true']) ** 2)
print(f"Test MSE for {input_filename}: {mse_test:.4f}")

# Calculate CI bounds
upper_bound = df_test['means'] + (1.96 * df_test['stds'])
lower_bound = df_test['means'] - (1.96 * df_test['stds'])

# 4. PLOTTING
plt.figure(figsize=(10, 6))

# A. Plot TEST set (Line + Confidence Interval)
plt.plot(df_test['true'], df_test['means'], label='Test Predictions (Mean)', color='blue', linewidth=2)
plt.fill_between(df_test['true'], lower_bound, upper_bound, color='blue', alpha=0.2, label='95% Confidence Interval')

# B. Plot TRAIN set (Scatter Points)
plt.scatter(df_train['true'], df_train['means'], label='Train Data', color='green', alpha=0.5, s=20, zorder=3)

# C. Plot Ideal Reference Line
min_val = min(df['true'].min(), df['means'].min())
max_val = max(df['true'].max(), df['means'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit')

# Formatting
plt.xlabel('True Values (Actual)')
plt.ylabel('Predicted Values (Mean)')
plt.title(f'Bayesian Linear Regression\n({input_filename})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(output_filename, dpi=300)
print(f"Plot successfully saved to {output_filename}")
