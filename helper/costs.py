import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curves(filename):
    # 1. Validate file exists
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        sys.exit(1)

    # 2. Load the CSV data
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # 3. Setup the plot
    plt.figure(figsize=(10, 6))
    
    # 4. Plot each column found in the CSV
    # This automatically handles 'poly cost', 'gaus cost', and 'sigm cost' headers
    for column in df.columns:
        plt.plot(df[column], label=column, linewidth=2, marker='o', markersize=4, markevery=5)

    # 5. Formatting for the Assessment Report
    plt.title('Logistic Regression Convergence: Basis Function Comparison', fontsize=14)
    plt.xlabel('Iterations (recorded every 10 epochs)', fontsize=12)
    plt.ylabel('Cross-Entropy Cost', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    # 6. Save based on input name
    output_image = filename.replace('.csv', '.png')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Successfully generated plot: {output_image}")
    plt.show()

if __name__ == "__main__":
    # Check if a filename was provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python plot_cost.py <filename.csv>")
    else:
        plot_learning_curves(sys.argv[1])