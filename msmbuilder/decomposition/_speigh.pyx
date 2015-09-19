from __future__ import print_function

from libc.math cimport sqrt, fabs

import numpy as np
import scipy.linalg
import scipy.special
from quadprog import solve_qp

include "cy_blas.pyx"

cdef double TAU_NEAR_ZERO_CUTOFF = 1e-6

cdef double max(double a, double b):
    if a < b:
        return b
    return a

cdef double sign(double x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


def speigh(double[:, ::1] A, double[:, ::1] B, double rho, v_init=None,
           double eps=1e-6, double tol=1e-8, tau=None, int maxiter=10000,
           int max_nc=100, greedy=True, verbose=False, return_x_f=False):
    """Find a sparse approximate generalized eigenpair.

    The generalized eigenvalue equation, :math:`Av = lambda Bv`,
    can be expressed as a variational optimization ::
    :math:`max_{x} x^T A x  s.t. x^T B x = 1`. We can search for sparse
    approximate eigenvectors then by adding a penalty to the optimization.
    This function solves an approximation to::

    max_{x}   x^T A x - \rho ||x||_0

        s.t.      x^T B x <= 1

    Where `||x||_0` is the number of nonzero elements in `x`. Note that
    because of the ||x||_0 term, that problem is NP-hard. Here, we replace
    the ||x||_0 term with

    rho * \sum_i^N \frac{\log(1 + |x_i|/eps)}{1 + 1/eps}

    which converges to ||x||_0 in the limit that eps goes to zero. This
    formulation can then be written as a d.c. (difference of convex) program
    and solved efficiently. The algorithm is due to [1], and is written
    on page 15 of the paper.

    Parameters
    ----------
    A : np.ndarray, shape=(N, N)
        A is symmetric matrix, the left-hand-side of the eigenvalue equation.
    B : np.ndarray, shape=(N, N)
        B is a positive semidefinite matrix, the right-hand-side of the
        eigenvalue equation.
    rho : float
        Regularization strength. Larger values for rho will lead to more sparse
        solutions.
    v_init : np.ndarray, shape=(N,), optional
        Initial guess for the eigenvector. This should probably be computed by
        running the standard generalized eigensolver first. If not supplied,
        we just use a vector of all ones.
    eps : float
        Small number, used in the approximation to the L0. Smaller is better
        (closer to L0), but trickier from a numerical standpoint and can lead
        to the solver complaining when it gets too small.
    tol : float
        Convergence criteria for the eigensolver.
    tau : float
        Should be the maximum of 0 and the negtion of smallest eigenvalue
        of A, ``tau=max(0, -lambda_min(A))``. If not supplied, the smallest
        eigenvalue of A will have to be computed.
    maxiter : int
        Maximum number of iterations.
    max_nc
        Maximum number of iterations without any change in the sparsity
        pattern
    return_x_f : bool, optional
        Also return the final iterate.

    Returns
    -------
    u : float
        The approximate eigenvalue.
    v_final : np.ndarray, shape=(N,)
        The sparse approximate eigenvector
    x_f : np.ndarray, shape=(N,), optional
        The sparse approximate eigenvector, before variational renormalization
        returned only in ``return_x_f = True``.

    References
    ----------
    ..[1] Sriperumbudur, Bharath K., David A. Torres, and Gert RG Lanckriet.
    "A majorization-minimization approach to the sparse generalized eigenvalue
    problem." Machine learning 85.1-2 (2011): 3-39.

    """
    cdef int i, j
    cdef int N = A.shape[0]

    if tau is None:
        tau = max(0, -np.min(scipy.linalg.eigvalsh(A)))


    cdef double rho_e = rho / scipy.special.log1p(1.0/eps)
    cdef double[::1] b = np.ascontiguousarray(np.diag(B))
    cdef int B_is_diagonal = np.all(np.diag(b) == B)



    if v_init is None:
        x = np.ones(N)
    else:
        x = np.array(v_init, copy=True)
    Ax = np.zeros(N)
    absAx = np.zeros(N)
    w = np.zeros(N)
    gamma = np.zeros(N)

    if tau < TAU_NEAR_ZERO_CUTOFF:
        if B_is_diagonal:
            print('Path [1]: tau=0, diagonal B')
            x, _ = _speigh_path_1(A, b, x, eps, rho_e, maxiter, tol)
        else:
            print('Path [2]: tau=0, general B')
            x, _ = _speigh_path_2(A, B, x, eps, rho_e, maxiter, tol)

    # Proposition 1 and the "variational renormalization" described in [1].
    # Use the sparsity pattern in 'x', but ignore the loadings and rerun an
    # unconstrained GEV problem on the submatrices determined by the nonzero
    # entries in our optimized x
    mask = (np.abs(x) > tol)
    grid = np.ix_(mask, mask)
    Ak, Bk = np.asarray(A)[grid], np.asarray(B)[grid]  # form the submatrices

    if len(Ak) == 0:
        u, v = 0, np.zeros(N)
    elif len(Ak) == 1:
        v = np.zeros(N)
        v[mask] = 1.0
        u = Ak[0,0] / Bk[0,0]
    else:
        gevals, gevecs = scipy.linalg.eigh(
            Ak, Bk, eigvals=(Ak.shape[0]-2, Ak.shape[0]-1))
        # Usually slower to use sparse linear algebra here
        # gevals, gevecs = scipy.sparse.linalg.eigsh(
        #     A=Ak, M=Bk, k=1, v0=x[mask], which='LA')
        u = gevals[-1]
        v = np.zeros(N)
        v[mask] = gevecs[:, -1]
    print('\nRenormalized sparse eigenvector:\n', v)

    if return_x_f:
        return u, v, x
    return u, v


cdef _speigh_path_1(double[:, ::1] A, double[::1] b, double[::1] x, double eps, double rho_e, int maxiter, double tol):
    cdef int i
    cdef int N = len(A)
    cdef double[::1] w = np.empty(N)
    cdef double[::1] Ax = np.empty(N)
    cdef double[::1] absAx = np.empty(N)
    cdef double[::1] gamma = np.empty(N)
    cdef double sum_gamma_over_b
    cdef double rq = np.inf
    cdef double old_rq = np.inf

    for i in range(maxiter):
        cdgemv_N(A, x, Ax)
        # check for absolute change in the rayleigh quotient
        old_rq, rq = rq, np.dot(Ax, x) / np.dot(np.multiply(b, x), x)
        # print('rq', rq)
        if fabs(rq - old_rq) < tol or np.isnan(rq):
            break

        sum_gamma_over_b = 0
        for j in range(N):
            w_j = 1.0 / (fabs(x[j]) + eps)
            gamma[j] = max(fabs(Ax[j]) - (rho_e/2) * w_j, 0)
            sum_gamma_over_b += gamma[j] / b[j]

        for j in range(N):
            x[j] = gamma[j] * sign(Ax[j]) / (b[j] * sum_gamma_over_b)

    return x, i


cdef _speigh_path_2(double[:, ::1] A, double[:, ::1] B, double[::1] x, double eps, double rho_e, int maxiter, double tol):
    cdef int i
    cdef int N = len(A)
    cdef double[::1] Ax = np.empty(N)
    cdef double[::1] gamma = np.empty(N)
    cdef double[::1] s = np.empty(N)
    cdef double[::1] zeros = np.zeros(N)
    cdef double w_j
    cdef double rq = np.inf
    cdef double old_rq = np.inf
    cdef double[:, ::1] eye = np.eye(N)

    for i in range(maxiter):
        cdgemv_N(A, x, Ax)
        # check for absolute change in the rayleigh quotient
        old_rq, rq = rq, np.dot(Ax, x) / np.dot(np.dot(B, x), x)
        # print('rq', rq)
        if fabs(rq - old_rq) < tol:
            break
        for j in range(N):
            w_j = 1.0 / (fabs(x[j]) + eps)
            gamma[j] = fabs(Ax[j]) - (rho_e/2) * w_j
            s[j]  = sign(Ax[j])
        S = np.diag(s)
        SBSi = scipy.linalg.pinv(np.dot(S, B).dot(S))
        # solve QP on line 20 of algorithm 1
        soln, val = solve_qp(SBSi, zeros, eye, gamma)[0:2]
        x = np.dot(S, SBSi).dot(soln) / sqrt(2*val)

    return x, i
