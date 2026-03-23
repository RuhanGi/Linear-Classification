import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

def plot_model_results(pred_file, weight_file):
    # 1. Validate files exist
    if not os.path.exists(pred_file):
        print(f"Error: Could not find {pred_file}")
        sys.exit(1)
    if not os.path.exists(weight_file):
        print(f"Error: Could not find {weight_file}")
        sys.exit(1)

    # Extract model name from filename (e.g., 'gda_predictions.csv' -> 'GDA')
    model_name = os.path.basename(pred_file).split('_')[0].upper()

    # 2. Load the exported C++ data
    df = pd.read_csv(pred_file)
    w = pd.read_csv(weight_file)['weight_value'].values

    X1 = df['x1'].values
    X2 = df['x2'].values
    y_true = df['actual'].values
    y_pred = df['predicted'].values

    # 3. Setup the Figure (1x2 Grid)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- LEFT PLOT: Decision Boundary ---
    ax1 = axes[0]
    x_min, x_max = X1.min() - 0.5, X1.max() + 0.5
    y_min, y_max = X2.min() - 0.5, X2.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Calculate boundary Z = w0 + w1*x1 + w2*x2
    Z = w[0] + w[1] * xx + w[2] * yy
    Z = Z >= 0 

    # Explicitly map 0 to pastel red, and 1 to pastel blue
    cmap_bg = ListedColormap(['#9999ff', '#ff9999']) 
    ax1.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)

    ax1.scatter(X1[y_true == 0], X2[y_true == 0], c='red', edgecolors='k', label='Class 0')
    ax1.scatter(X1[y_true == 1], X2[y_true == 1], c='blue', edgecolors='k', label='Class 1')
    
    ax1.set_title(f'Decision Boundary')
    ax1.set_xlabel('Normalized Feature 1')
    ax1.set_ylabel('Normalized Feature 2')
    ax1.legend(loc='best')

    # --- RIGHT PLOT: Confusion Matrix ---
    ax2 = axes[1]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False,
                xticklabels=['Pred 1', 'Pred 0'], 
                yticklabels=['Actual 1', 'Actual 0'],
                annot_kws={"size": 16})
    
    ax2.set_title(f'Confusion Matrix')

    # 4. Save and Show
    output_filename = f'{model_name.lower()}_results.png'
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Saved plot as {output_filename}")
    plt.show()

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Check if the user passed exactly 2 arguments (plus the script name)
    if len(sys.argv) != 3:
        print("Usage: python plot_results.py <predictions.csv> <weights.csv>")
        sys.exit(1)

    pred_csv = sys.argv[1]
    weight_csv = sys.argv[2]
    
    print(f"Generating plots for {pred_csv} and {weight_csv}...")
    plot_model_results(pred_csv, weight_csv)