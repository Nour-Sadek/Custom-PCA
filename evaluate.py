import numpy as np
from custom_pca import CustomPCAClass
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Creating the dataset
np.random.seed(0)
X = np.random.rand(100, 5)

custom_pca = CustomPCAClass(X)
sklearn_pca = PCA()

# Get the transformed X matrix from each implementation
X_transformed_sklearn = sklearn_pca.fit_transform(X)
X_transformed_custom = custom_pca.pca_transformed_matrix

# Compare the covariance matrices and eigenvalues
print("Does the custom implementation give the same covariance matrix as the Scikit-learn implementation?")
print(np.allclose(sklearn_pca.get_covariance(), custom_pca.covariance_matrix))
print()
print("Does the custom implementation give the same eigenvalues as the Scikit-learn implementation?")
print(np.allclose(sklearn_pca.explained_variance_, custom_pca.eigenvalues))
print()

# Plot both sets of points
plt.figure(figsize=(8, 6))

# PCA points for each implementation
plt.scatter(X_transformed_sklearn[:, 0], X_transformed_sklearn[:, 1], c='blue', alpha=0.7, label='Scikit-learn PCA')
plt.scatter(X_transformed_custom[:, 0], X_transformed_custom[:, 1], c='orange', alpha=0.5, label='Custom PCA')

plt.title('PCA Plot: PC1 vs PC2 (Comparison)')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend()
plt.show()
