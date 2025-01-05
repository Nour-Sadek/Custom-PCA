import numpy as np
from scipy.sparse.linalg import eigsh


class CustomPCAClass:

    """A custom PCA class that does PCA analysis on a numpy 2d array of the shape n_observations x n_features

    === Private Attributes ===
    _X: The given numpy 2d array to perform PCA on
    _covariance_matrix: The covariance matrix of <__X>
    _eigenvalues: The eigenvalues in decreasing order of the <_covariance_matrix>
    _transformed_X: The transformed <__X> matrix after performing PCA analysis

    """

    _X: np.ndarray
    _X_centered: np.ndarray
    _covariance_matrix: np.ndarray
    _eigenvalues: np.ndarray
    _eigenvectors: np.ndarray
    _transformed_X: np.ndarray

    def __init__(self, x: np.ndarray, centered: bool = False) -> None:
        """Initialize a new CustomPCAClass with <X> as the matrix."""
        self._X = x
        if not centered:
            self._X_centered = self._center_data()
        else:
            self._X_centered = self._X
        self._covariance_matrix = self._compute_covariance_matrix()
        self._eigenvalues, self._eigenvectors = self._compute_eigenvalues_eigenvectors()
        self._transformed_X = self._perform_pca()

    @property
    def covariance_matrix(self) -> np.ndarray:
        return self._covariance_matrix

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        return self._eigenvectors

    @property
    def pca_transformed_matrix(self) -> np.ndarray:
        return self._transformed_X

    def _center_data(self) -> np.ndarray:
        """Return a modified numpy array where <__X> is centered"""
        cols_mean = np.mean(self._X, axis=0)
        cols_mean_mat = cols_mean * np.ones((self._X.shape[0], self._X.shape[1]))
        centered_data = self._X - cols_mean_mat
        return centered_data

    def _compute_covariance_matrix(self) -> np.ndarray:
        """Return the covariance matrix of the centered <_X> array"""
        covar_mat = self._X_centered.transpose() @ self._X_centered
        covar_mat = covar_mat / (self._X_centered.shape[0] - 1)
        return covar_mat

    def _compute_eigenvalues_eigenvectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a tuple of (eigenvalues, eigenvectors) of <_covariance_matrix> in decreasing order of eigenvalues"""
        # eigenvectors returned all have a norm of 1 so no need to create a projection matrix
        # that divides each eigenvector by its L2-norm
        eigenvalues, eigenvectors = eigsh(self._covariance_matrix, k=self._covariance_matrix.shape[0])
        # Change order from increasing to decreasing order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        return eigenvalues, eigenvectors

    def _perform_pca(self) -> np.ndarray:
        return self._X_centered @ self._eigenvectors
