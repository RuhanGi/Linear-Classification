import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

if len(sys.argv) < 2:
    print("Error: No input CSV filename provided.")
    print("Usage: python plot_heatmap.py posterior_weights.csv")
    sys.exit(1)

input_filename = sys.argv[1]
base_name, _ = os.path.splitext(input_filename)
output_filename = f"{base_name}_heatmap.png"

# 1. Load the data
df = pd.read_csv(input_filename)

# Rebuild Full mN Vector
mN_df = df[df['type'] == 'mN'].sort_values('idx1')
mN_full = mN_df['val'].values
M = len(mN_full)

# Rebuild Full SN Matrix
SN_df = df[df['type'] == 'SN']
SN_full = np.zeros((M, M))
for _, row in SN_df.iterrows():
    SN_full[int(row['idx1']), int(row['idx2'])] = row['val']

# 2. Slice for the 2D Heatmap (w0 and w1 only)
mN_2d = mN_full[0:2]
SN_2d = SN_full[0:2, 0:2]

# 3. Setup the Single Plot
plt.figure(figsize=(6, 5))

# Create grid boundaries dynamically around the mean
w0_min, w0_max = mN_2d[0] - 1.5, mN_2d[0] + 1.5
w1_min, w1_max = mN_2d[1] - 1.5, mN_2d[1] + 1.5

w0_grid, w1_grid = np.mgrid[w0_min:w0_max:100j, w1_min:w1_max:100j]
pos = np.dstack((w0_grid, w1_grid))

# Calculate the Gaussian PDF
posterior_dist = multivariate_normal(mean=mN_2d, cov=SN_2d)
pdf_values = posterior_dist.pdf(pos)

# Draw the Heatmap
plt.contourf(w0_grid, w1_grid, pdf_values, levels=50, cmap='jet')

# Mark the exact Mean/MAP with a white cross
plt.plot(mN_2d[0], mN_2d[1], 'w+', markersize=12, markeredgewidth=2, label='Posterior Mean')

# Formatting
plt.xlabel('$w_0$ (Intercept)', fontsize=12)
plt.ylabel('$w_1$ (First Feature)', fontsize=12)
plt.title('Final Posterior in Weight Space', fontsize=14)
plt.xlim([w0_min, w0_max])
plt.ylim([w1_min, w1_max])

# Save and Show
plt.tight_layout()
plt.savefig(output_filename, dpi=300)
print(f"Heatmap successfully saved to {output_filename}")