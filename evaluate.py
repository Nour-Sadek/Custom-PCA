import numpy as np
from custom_pca import CustomPCAClass
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


"""Testing the custom PCA function on a random dataset where observations are not expected to cluster"""

# Creating the dataset
np.random.seed(0)
X = np.random.rand(100, 5)

custom_pca = CustomPCAClass(X)
sklearn_pca = PCA()

# Get the transformed X matrix from each implementation
X_transformed_sklearn = sklearn_pca.fit_transform(X)
X_transformed_custom = custom_pca.pca_transformed_matrix

# Compare the covariance matrices and eigenvalues
print("On the randomly generated dataset:\n")
print("Does the custom implementation give the same covariance matrix as the Scikit-learn implementation?")
print(np.allclose(sklearn_pca.get_covariance(), custom_pca.covariance_matrix))
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


"""Testing the custom PCA function on the available Scikit-learn wine dataset where observations are expected to 
cluster based on their target classification"""

# Loading the dataset
wine = load_wine()
wine_data = wine.data
labels = wine.target

# Normalizing the values by standardization (range of values differs between features)
standard_scaler = StandardScaler()
wine_data_scaled = standard_scaler.fit_transform(wine_data)

wine_custom_pca = CustomPCAClass(wine_data_scaled, centered=True)
wine_sklearn_pca = PCA()

# Get the transformed X matrix from each implementation
wine_transformed_sklearn = wine_sklearn_pca.fit_transform(wine_data_scaled)
wine_transformed_custom = wine_custom_pca.pca_transformed_matrix

# Compare the covariance matrices and eigenvalues
print("On the labeled wine dataset:\n")
print("Does the custom implementation give the same covariance matrix as the Scikit-learn implementation?")
print(np.allclose(wine_sklearn_pca.get_covariance(), wine_custom_pca.covariance_matrix))
print("Does the custom implementation give the same eigenvalues as the Scikit-learn implementation?")
print(np.allclose(wine_sklearn_pca.explained_variance_, wine_custom_pca.eigenvalues))
print()

# Plotting PC1-v-PC2 for both implementations side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Scatterplot for Scikit-learn PCA
for label, color in zip(np.unique(labels), ['r', 'g', 'b']):
    # Scatterplot for Scikit-learn PCA
    axes[0].scatter(
        wine_transformed_sklearn[labels == label, 0],
        wine_transformed_sklearn[labels == label, 1],
        label=label
    )
    # Scatterplot for custom PCA
    axes[1].scatter(
        wine_transformed_custom[labels == label, 0],
        wine_transformed_custom[labels == label, 1],
        label=label
    )

# Graph labeling for Scikit-learn PCA figure
axes[0].set_title("Scikit-learn PCA")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()

# Graph labeling for custom PCA figure
axes[1].set_title("Custom PCA")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()

plt.tight_layout()
plt.show()
