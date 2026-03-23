import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_binary_data(n_samples=200, n_features=2, n_informative=2, filename="synthetic_data.csv"):
    """
    Generates synthetic binary classification data.
    
    n_samples: Total number of rows
    n_features: Total number of columns (N)
    n_informative: Number of features actually used to build the boundary
    """
    # Generate the data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.01, # Adds a tiny bit of noise/overlap
        class_sep=1.5, # Larger value = easier to separate
        random_state=42
    )

    # Create dynamic headers: x1, x2, ..., xN
    headers = [f"x{i+1}" for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=headers)
    df['target'] = y

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Successfully generated {filename} with {n_features} features.")

# Example: Generate a 2D dataset for easy plotting
generate_binary_data(n_samples=300, n_features=2, n_informative=2)