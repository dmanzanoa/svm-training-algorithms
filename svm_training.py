"""SVM Training Algorithms Module
==================================

This module implements soft‑margin Support Vector Machine (SVM) classifiers
using both the *primal* and *dual* optimisation formulations. It also
includes a simple kernel interface allowing linear and non‑linear kernels
such as polynomial, radial basis function (RBF) and inverse multiquadric
kernels. The code has been refactored from a Jupyter notebook into a
plain Python module so that it can be imported and reused in other
projects or executed as a standalone script for demonstration purposes.

The primary classes are:

* :class:`SVM` – An abstract base class specifying the common interface
  for SVM classifiers. It provides methods for fitting, predicting,
  evaluating and optionally visualising two‑dimensional decision
  boundaries.
* :class:`PrimalSVM` – Implements a soft‑margin SVM solved via the
  primal objective using stochastic gradient descent. This is the
  traditional linear SVM where the weight vector is optimised directly.
* :class:`DualSVM` – Implements a soft‑margin SVM solved via the dual
  objective using projected gradient ascent on the Lagrange multipliers
  (alphas). This formulation allows the use of kernel functions to
  capture non‑linear decision boundaries.
* :class:`Kernel` – a simple kernel factory supporting linear,
  polynomial, RBF and inverse multiquadric kernels. Custom kernels can
  be provided by passing a callable to :class:`DualSVM`.

Examples
--------
The module can be executed directly to see how the algorithms perform on
synthetic data:

.. code-block:: bash

    python svm_training.py --demo

This will generate a toy binary classification dataset, train both the
primal and dual SVMs and print their accuracies. It can also show
decision boundaries if matplotlib is available and two‑dimensional
features are used.

Classes and Functions
---------------------
``SVM``
    Base class for SVM classifiers.
``PrimalSVM``
    Soft‑margin SVM using the primal formulation.
``DualSVM``
    Soft‑margin SVM using the dual formulation with kernel support.
``Kernel``
    Factory for standard kernel functions.

Author
------
This script was derived from an academic notebook and refactored into a
standalone module as part of a project demonstrating SVM training
algorithms. It is intended to be concise and easy to follow for those
interested in machine learning implementation details.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Callable, Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


class SVM(abc.ABC):
    """Abstract base class for soft‑margin SVM classifiers.

    Subclasses must implement a ``fit`` method that learns model
    parameters from the training data. The base class provides common
    functionality for prediction, accuracy evaluation and optional
    visualisation of the decision boundary on two‑dimensional data.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, iterations: int = 100) -> None:
        """Fit the SVM model on the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vectors.
        y : ndarray of shape (n_samples,)
            Target labels, expected to be +1 or −1.
        iterations : int, default=100
            Number of training iterations. For stochastic optimisers this
            corresponds to epochs over the data.
        """
        ...

    def predict(self, test_X: np.ndarray) -> np.ndarray:
        """Predict labels for the provided test data.

        Parameters
        ----------
        test_X : ndarray of shape (n_samples, n_features)
            Test samples to classify.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels, +1 or −1.
        """
        assert self._is_fitted, "You must fit the SVM before predicting."
        return self._predict_internal(test_X)

    @abc.abstractmethod
    def _predict_internal(self, test_X: np.ndarray) -> np.ndarray:
        """Internal prediction logic implemented by subclasses."""
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the classification accuracy of the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data samples.
        y : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Accuracy between 0 and 1.
        """
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def visualize(self, X: np.ndarray, y: np.ndarray, title: str = "Decision boundary") -> None:
        """Visualise the decision boundary for two‑dimensional data.

        If the data has more than two features this function will raise a
        ``ValueError``. The decision surface is plotted by sampling
        across a grid and evaluating the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            Two‑dimensional input data.
        y : ndarray of shape (n_samples,)
            Corresponding labels (±1).
        title : str, default="Decision boundary"
            Title for the plot.
        """
        if X.shape[1] != 2:
            raise ValueError("Visualization only supported for 2D features.")

        # Create a meshgrid over the input domain
        x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
        y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid).reshape(xx.shape)

        plt.figure(figsize=(6, 4))
        plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1, alpha=0.8)
        plt.scatter(X[y == -1, 0], X[y == -1, 1], c="blue", label="-1")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c="red", label="+1")
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()


class PrimalSVM(SVM):
    """Soft‑margin SVM solved using the primal objective.

    This implementation uses a simple stochastic gradient descent
    optimiser to learn the weight vector ``w`` and bias ``b``. It is
    suitable for linearly separable or nearly separable datasets. The
    hinge loss is used and a regularisation parameter ``lambda0``
    controls the trade‑off between the margin and slack variables.

    Parameters
    ----------
    eta : float
        Learning rate for gradient descent.
    lambda0 : float
        Regularisation strength. Must be strictly positive.
    """

    def __init__(self, eta: float, lambda0: float) -> None:
        super().__init__()
        if lambda0 <= 0:
            raise ValueError("lambda0 must be positive.")
        self.eta = eta
        self.lambda0 = lambda0
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, iterations: int = 100) -> None:
        super().fit(X, y, iterations)
        n_samples, n_features = X.shape
        # Initialize weights and bias randomly for symmetry breaking
        rng = np.random.default_rng(42)
        self.w = rng.standard_normal(n_features)
        self.b = float(rng.standard_normal())

        for _ in range(iterations):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) < 1
                if condition:
                    # Gradient for misclassified or within‑margin points
                    grad_w = self.lambda0 * self.w - np.dot(x_i, y[idx])
                    self.w -= self.eta * grad_w
                    self.b += self.eta * y[idx]
                else:
                    # Only apply regularisation term to the weights
                    self.w -= self.eta * self.lambda0 * self.w
        self._is_fitted = True

    def _predict_internal(self, test_X: np.ndarray) -> np.ndarray:
        assert self.w is not None, "Model must be fitted before predicting."
        linear_output = np.dot(test_X, self.w) + self.b
        return np.sign(linear_output)

    def get_weights(self) -> np.ndarray:
        """Return the learned weight vector for inspection."""
        assert self.w is not None, "Model has not been fitted."
        return self.w

    def get_bias(self) -> float:
        """Return the learned bias term."""
        return self.b


class DualSVM(SVM):
    """Soft‑margin SVM solved using the dual objective with kernels.

    The dual formulation expresses the SVM optimisation problem in terms
    of Lagrange multipliers (``alphas``). This implementation performs
    projected stochastic gradient ascent on the dual variables. It
    supports arbitrary kernel functions via a ``kernel`` argument. For a
    linear kernel the method also computes the primal weight vector for
    convenience.

    Parameters
    ----------
    eta : float
        Learning rate for the dual optimizer.
    C : float
        Regularisation parameter controlling the margin slack.
    kernel : callable, optional
        A function ``k(u, v)`` returning the kernel value between two
        input vectors. If ``None``, the linear kernel is used.
    """

    def __init__(self, eta: float, C: float, kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None) -> None:
        super().__init__()
        if C <= 0:
            raise ValueError("C must be positive.")
        self.eta = eta
        self.C = C
        self.kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] = (
            kernel if kernel is not None else lambda u, v: u @ v.T
        )
        # Model parameters
        self.alphas: Optional[np.ndarray] = None
        self.b: float = 0.0
        # Cache training data
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None  # Only for linear kernel

    def fit(self, X: np.ndarray, y: np.ndarray, iterations: int = 100) -> None:
        super().fit(X, y, iterations)
        n_samples, n_features = X.shape
        self.X = X
        self.y = y
        # Initialise alphas at zero
        self.alphas = np.zeros(n_samples)

        for _ in range(iterations):
            for i in range(n_samples):
                # Gradient of the dual objective with respect to alpha_i
                kernel_vals = self.kernel(self.X, self.X[i])  # shape (n_samples,)
                dL_dalpha = 1 - y[i] * np.dot(self.y * self.alphas, kernel_vals)
                # Update with projection onto [0, C]
                self.alphas[i] = np.clip(self.alphas[i] + self.eta * dL_dalpha, 0, self.C)

        # Compute primal weights for linear kernel only
        if kernel is None or self.kernel == (lambda u, v: u @ v.T):
            self.w = self._compute_primal_weights()
        # Compute bias term
        self.b = self._compute_bias()
        self._is_fitted = True

    def _predict_internal(self, test_X: np.ndarray) -> np.ndarray:
        assert self.alphas is not None, "Model must be fitted before predicting."
        assert self.X is not None and self.y is not None, "Training data not cached."
        kernel_matrix = self.kernel(self.X, test_X)  # shape (n_samples, n_test)
        preds = (self.alphas * self.y) @ kernel_matrix + self.b
        return np.sign(preds)

    def _compute_primal_weights(self) -> np.ndarray:
        """Compute the primal weight vector for a linear kernel.

        Returns
        -------
        ndarray
            Weight vector of shape (n_features,).
        """
        assert self.alphas is not None and self.X is not None and self.y is not None
        return np.sum((self.alphas * self.y)[:, None] * self.X, axis=0)

    def _compute_bias(self) -> float:
        """Compute the bias term using support vectors.

        The bias is averaged over all support vectors ``i`` such that
        ``0 < alpha_i < C``. If no such support vectors exist, the bias
        defaults to zero.
        """
        assert self.alphas is not None and self.X is not None and self.y is not None
        n_samples = len(self.alphas)
        bias_total = 0.0
        count = 0
        for i in range(n_samples):
            if 0 < self.alphas[i] < self.C:
                tmp = self.y[i]
                kernel_vals = self.kernel(self.X[i], self.X)
                tmp -= np.sum(self.alphas * self.y * kernel_vals)
                bias_total += tmp
                count += 1
        return bias_total / count if count > 0 else 0.0

    def get_support_vectors(self) -> np.ndarray:
        """Return the indices of support vectors, i.e. those with 0 < alpha < C."""
        assert self.alphas is not None
        return np.where((self.alphas > 0) & (self.alphas < self.C))[0]

    def get_alpha_equals_C(self) -> np.ndarray:
        """Return the indices of training points where alpha equals C."""
        assert self.alphas is not None
        return np.where(self.alphas == self.C)[0]


class Kernel:
    """Factory for common kernel functions used with the dual SVM.

    Parameters
    ----------
    kernel_type : str
        One of ``{"linear", "poly", "rbf", "imq"}``. Defaults to
        ``"linear"`` if unspecified.
    degree : int, optional
        Degree of the polynomial kernel (for ``"poly"``).
    offset : float, optional
        Offset term added to the polynomial kernel (for ``"poly"``).
    gamma : float, optional
        Gamma parameter for the RBF kernel (for ``"rbf"``).
    c : float, optional
        Parameter ``c`` for the inverse multiquadric (IMQ) kernel (for
        ``"imq"``).
    """

    def __init__(self, kernel_type: str = "linear", **kwargs) -> None:
        kernel_type = kernel_type.lower()
        if kernel_type == "linear":
            self.kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] = self.linear_kernel
        elif kernel_type == "poly":
            self.degree: int = kwargs.get("degree", 3)
            self.offset: float = kwargs.get("offset", 1.0)
            self.kernel = self.poly_kernel
        elif kernel_type == "rbf":
            self.gamma: float = kwargs.get("gamma", 1.0)
            self.kernel = self.rbf_kernel
        elif kernel_type == "imq":
            self.c: float = kwargs.get("c", 1.0)
            self.kernel = self.imq_kernel
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

    def __call__(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.kernel(u, v)

    @staticmethod
    def linear_kernel(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u @ v.T

    def poly_kernel(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (u @ v.T + self.offset) ** self.degree

    def rbf_kernel(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        dists = norm(u[:, None] - v, axis=2) ** 2
        return np.exp(-self.gamma * dists)

    def imq_kernel(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(u)
        v = np.atleast_2d(v)
        dists = norm(u[:, None] - v, axis=2) ** 2
        return 1.0 / np.sqrt(dists + self.c)


def _demo() -> None:
    """Demonstration of the SVM algorithms on synthetic data.

    Generates a simple two‑dimensional classification dataset, trains both
    the primal and dual SVMs and prints their accuracies. If matplotlib
    is available, it also visualises the decision boundaries.
    """
    from sklearn.datasets import make_classification

    # Create toy data (linearly separable with two informative features)
    X, y = make_classification(
        n_samples=60, n_features=2, n_informative=2, n_redundant=0, random_state=42
    )
    # Convert labels from {0, 1} to {-±, +1}
    y = np.where(y == 0, -1, 1)

    # Train primal SVM
    psvm = PrimalSVM(eta=0.1, lambda0=0.1)
    psvm.fit(X, y, iterations=100)
    print(f"Primal SVM accuracy: {psvm.evaluate(X, y):.3f}")
    try:
        psvm.visualize(X, y, title="Primal SVM Decision Boundary")
    except ValueError:
        pass

    # Train dual SVM with RBF kernel
    rbf = Kernel(kernel_type="rbf", gamma=0.5)
    dsvm = DualSVM(eta=0.1, C=10.0, kernel=rbf)
    dsvm.fit(X, y, iterations=100)
    print(f"Dual SVM (RBF kernel) accuracy: {dsvm.evaluate(X, y):.3f}")
    try:
        dsvm.visualize(X, y, title="Dual SVM Decision Boundary (RBF)")
    except ValueError:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demonstration for SVM training algorithms")
    parser.add_argument("--demo", action="store_true", help="Run a demo on synthetic data")
    args = parser.parse_args()

    if args.demo:
        _demo()
