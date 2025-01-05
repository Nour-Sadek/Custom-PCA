# Custom PCA Implementation

This repository compares a custom PCA implementation to Scikit-learn's PCA function. 
The comparison is performed on two datasets:

- __Random Dataset__: Where observations are not expected to form clusters.
- __Wine Dataset__: Where observations are expected to cluster based on class labels.

# Usage

Run the evaluation script:

    python evaluate.py

Expected outputs:
- Comparison of the covariance matrix and eigenvalues of both implementations
- Scatter plots of PC1 vs. PC2 for both implementations on both the random and wine datasets

# Results

- The eigenvalues and covariance matrix from the custom PCA match Scikit-learn's PCA closely.
- Visualizations:
  - On the random dataset, both PCA methods produce similar random scatter patterns.
  - On the wine dataset, both PCA methods reveal clustering consistent with the three wine classes, 
    with slight differences in alignment due to eigenvector orientation.

# Repository Structure

This repository contains:

    custom_pca.py: Implementation of the CustomPCAClass
    evaluate.py: Main script for performing PCA comparisons and generating plots
    requirements.txt: List of required Python packages

Python 3.12 version was used
