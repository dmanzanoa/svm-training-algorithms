# SVM Training Algorithms

This repository contains a concise Python implementation of soft‑margin
Support Vector Machines (SVMs) suitable for educational purposes and
small‑scale machine learning projects. The code provides both **primal**
and **dual** optimisation formulations and supports linear as well as
non‑linear decision boundaries via kernel functions. It was refactored
from a notebook into a modular script so that it can be imported as a
library or executed as a standalone example.

## Overview

The heart of the repository is `svm_training.py`, which defines the
following components:

| Component       | Description |
|-----------------|-------------|
| `SVM`           | An abstract base class specifying the interface for SVM classifiers, including `fit()`, `predict()`, `evaluate()` and 2‑D `visualize()` methods. |
| `PrimalSVM`     | Implements a soft‑margin SVM by directly optimising the weight vector and bias using stochastic gradient descent (primal formulation). |
| `DualSVM`       | Solves the soft‑margin SVM using the dual formulation with projected gradient ascent on the Lagrange multipliers. It supports arbitrary kernels. |
| `Kernel`        | A simple factory for common kernels (linear, polynomial, radial basis function (RBF) and inverse multiquadric (IMQ)). |

The module can be run in **demo mode** by executing:

```
python svm_training.py --demo
```

This generates a toy binary classification dataset, fits both the
primal and dual SVMs and prints their accuracies. If your data has two
features and `matplotlib` is installed, the script will also display
decision boundary plots.

## Installation

To install the required dependencies, create a virtual environment (optional) and install from `requirements.txt`:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The main dependencies are:

* `numpy` – numerical computations
* `matplotlib` – optional for visualisation
* `scikit-learn` – used in the demo to generate synthetic datasets

## Usage

You can import the classes from `svm_training.py` in your own code and
train SVM classifiers on any tabular dataset. The labels should be
encoded as +1 and −1. For example:

```python
from svm_training import PrimalSVM, DualSVM, Kernel
import numpy as np

# X: array of shape (n_samples, n_features)
# y: array of shape (n_samples,) with values +1 and −1

# Train a linear SVM using the primal form
primal = PrimalSVM(eta=0.1, lambda0=0.1)
primal.fit(X, y)
preds = primal.predict(X_test)

# Train a kernel SVM using the dual form with an RBF kernel
rbf_kernel = Kernel(kernel_type="rbf", gamma=0.5)
dual = DualSVM(eta=0.1, C=1.0, kernel=rbf_kernel)
dual.fit(X, y)
preds = dual.predict(X_test)
```

See the inline documentation in `svm_training.py` for more details on
the available parameters and methods.

## Background

This project demonstrates the core algorithms behind Support Vector
Machines. The primal and dual formulations showcase how the same
optimisation problem can be approached from different perspectives: the
primal SVM learns a separating hyperplane directly, while the dual SVM
optimises over Lagrange multipliers and opens the door to non‑linear
decision boundaries through kernels. These methods are widely used in
statistical learning and pattern recognition for classification tasks
such as handwriting recognition, bioinformatics and text categorisation.

## Licence

This repository is released under the MIT Licence. Feel free to use,
modify and distribute the code for educational or commercial purposes.
