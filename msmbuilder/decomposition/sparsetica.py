# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import
from six import PY2
import numpy as np
import scipy.linalg
from .tica import tICA
from ..utils import experimental, array2d
from ._speigh import speigh, scdeflate
from covar import cov_shrink

__all__ = ['SparseTICA']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class SparseTICA(tICA):
    """Sparse time-structure Independent Component Analysis (tICA).

    Linear dimensionality reduction which finds sparse linear combinations
    of the input features which decorrelate most slowly. These can be
    used for feature selection and/or dimensionality reduction.

    This model requires the additional python package `cvxpy`, which can be
    installed from `PyPI <https://pypi.python.org/pypi/cvxpy/>`_.

    .. warning::

        This model is currently  experimental, and may undergo significant
        changes or bug fixes in upcoming releases.

    Parameters
    ----------
    n_components : int
        Number of sparse tICs to find.
    lag_time : int
        Delay time forward or backward in the input data. The time-lagged
        correlations is computed between datas X[t] and X[t+lag_time].
    rho : positive float
        Regularization strength with controls the sparsity of the solutions.
        Higher values of rho gives more sparse tICS with nonozero loadings on
        fewer degrees of freedom. rho=0 corresponds to standard tICA.
    shrinkage : float, default=None
        The covariance shrinkage intensity (range 0-1). If shrinkage is not
        specified (the default) it is estimated using an analytic formula
        (the Rao-Blackwellized Ledoit-Wolf estimator) introduced in [5].
    weighted_transform : bool, default=False
        If True, weight the projections by the implied timescales, giving
        a quantity that has units [Time].
    epsilon : positive float, default=1e-6
        epsilon should be a number very close to zero, which is used to
        construct the approximation to the L_0 penality function. However,
        when it gets *too* close to zero, the solvers may report feasibility
        problems due to numerical stability issues. 1e-6 is a fairly good
        balance here.
    tolerance : positive float
        Convergence criteria for the sparse generalized eigensolver.
    maxiter : int
        Maximum number of iterations for the sparse generalized eigensolver.
    verbose : bool, default=False
        Print verbose information from the sparse generalized eigensolver.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        Components with maximum autocorrelation.
    offset_correlation_ : array-like, shape (n_features, n_features)
        Symmetric time-lagged correlation matrix, `C=E[(x_t)^T x_{t+lag}]`.
    eigenvalues_ : array-like, shape (n_features,)
        Psuedo-eigenvalues of the tICA generalized eigenproblem, in decreasing
        order.
    eigenvectors_ : array-like, shape (n_components, n_features)
        Sparse psuedo-eigenvectors of the tICA generalized eigenproblem. The
        vectors give a set of "directions" through configuration space along
        which the system relaxes towards equilibrium.
    means_ : array, shape (n_features,)
        The mean of the data along each feature
    n_observations_ : int
        Total number of data points fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
        online learning.
    n_sequences_ : int
        Total number of sequences fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
         online learning.
    timescales_ : array-like, shape (n_components,)
        The implied timescales of the tICA model, given by
        -offset / log(eigenvalues)

    See Also
    --------
    msmbuilder.decomposition.tICA

    References
    ----------
    .. [1] McGibbon, R. T. and V. S. Pande "Identification of sparse, slow
       reaction coordinates from molular dynamics simulations" In preparation.
    .. [1] Sriperumbudur, B. K., D. A. Torres, and G. R. Lanckriet.
       "A majorization-minimization approach to the sparse generalized eigenvalue
       problem." Machine learning 85.1-2 (2011): 3-39.
    .. [3] Mackey, L. "Deflation Methods for Sparse PCA." NIPS. Vol. 21. 2008.
    """

    def __init__(self, n_components, lag_time=1, rho=0.01,
                 weighted_transform=True, epsilon=1e-6, shrinkage=None,
                 tolerance=1e-6, maxiter=10000, verbose=False):
        super(SparseTICA, self).__init__(n_components, lag_time=lag_time,
            weighted_transform=weighted_transform)
        self.rho = rho
        self.epsilon = epsilon
        self.shrinkage = shrinkage
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.verbose = verbose

        self._sequences = []
        self._covariance_ = None

    def _fit(self, X):
        if self._initialized is False:
            self._sequences = []
        X = np.asarray(array2d(X), dtype=np.float64)
        self._sequences.append(X)
        super(SparseTICA, self)._fit(X)

    def _solve(self):
        if not self._is_dirty:
            return

        if self.rho <= 0:
            # if no sparse regularization, it's just regular tICA
            return super(SparseTICA, self)._solve()

        A = self.offset_correlation_
        B = self.covariance_

        self._eigenvalues_ = np.zeros((self.n_components))
        self._eigenvectors_ = np.zeros((self.n_features, self.n_components))

        for i in range(self.n_components):
            u, v = speigh(A, B, rho=self.rho, eps=self.epsilon,
                          tol=self.tolerance, maxiter=self.maxiter,
                          verbose=self.verbose)

            self._eigenvalues_[i] = u
            self._eigenvectors_[:, i] = v
            A = scdeflate(A, v)

        self._is_dirty = False

    def summarize(self):
        """Some summary information."""
        nonzeros = np.sum(np.abs(self.eigenvectors_) > 0, axis=0)
        active = '[%s]' % ', '.join(['%d/%d' % (n, self.n_features) for n in nonzeros[:5]])

        return """Sparse time-structure based Independent Components Analysis (tICA)
------------------------------------------------------------------
n_components        : {n_components}
gamma               : {gamma}
lag_time            : {lag_time}
weighted_transform  : {weighted_transform}
rho                 : {rho}
n_features          : {n_features}

Top 5 timescales :
{timescales}

Top 5 eigenvalues :
{eigenvalues}

Number of active degrees of freedom:
{active}
""".format(n_components=self.n_components, lag_time=self.lag_time, rho=self.rho,
           gamma=self.gamma, weighted_transform=self.weighted_transform,
           timescales=self.timescales_[:5], eigenvalues=self.eigenvalues_[:5],
           n_features=self.n_features, active=active)
