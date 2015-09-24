import numpy as np
import scipy.linalg
from msmbuilder.decomposition._speigh import speigh, scdeflate
from msmbuilder.decomposition._speigh import solve_cvxpy, solve_admm2



def rand_pos_semidef(n, seed=0):
    # random positive semidefinite matrix
    # http://stackoverflow.com/a/619406/1079728
    # (construct cholesky factor, and then return matrix)
    random = np.random.RandomState(seed)
    A = random.rand(n, n)
    B = np.dot(A, A.T)
    return B


def rand_sym(n, seed=0):
    # random symmetric
    random = np.random.RandomState(seed)
    A = random.randn(n, n)
    return A + A.T


def build_lowrank(n, v, seed=0):
    """Return n x n matrices (A, B) such that A is symmetric, B is positive
    definite, and the 1st generalized eigenvector of (A,B) is v, with an
    associated eigenvalue of 1. The other eigenvalues are 0.
    """
    random = np.random.RandomState(seed)
    # http://stackoverflow.com/a/27436614/1079728
    V = random.rand(n, n)
    V[:, 0] = v
    w = np.zeros(n)
    w[0] = 1

    A = scipy.linalg.inv(V.T).dot(np.diag(w)).dot(scipy.linalg.inv(V))
    B = scipy.linalg.inv(V.dot(V.T))
    return np.ascontiguousarray(A), np.ascontiguousarray(B)


def eigh(A, B=None):
    w, V = scipy.linalg.eigh(A, b=B)
    order = np.argsort(-w)
    w = w[order]
    V = V[:, order]
    return w, V


#######################################################


class Test_scdeflate(object):

    def test_1(self):
        n = 4
        A = rand_sym(n)

        w1, V1 = eigh(A)
        Ad = scdeflate(A, V1[:, 0])
        w2, V2 = eigh(Ad)

        self.assert_deflated(w1, V1, w2, V2)

    def test_2(self):
        n = 4
        A = rand_sym(n)
        B = rand_pos_semidef(n)
        w1, V1 = eigh(A, B)

        Ad = scdeflate(A, V1[:, 0])
        w2, V2 = eigh(Ad, B)

        self.assert_deflated(w1, V1, w2, V2)

    def assert_deflated(self, w1, V1, w2, V2):
        # the deflated matrix should have a one zero eigenvalue for the
        # vector that was deflated out.
        near_zero = (np.abs(w2) < 1e-10)
        assert np.sum(near_zero) == 1
        # the other eigenvalues should be unchanged
        np.testing.assert_array_almost_equal(
            w1[1:], w2[np.logical_not(near_zero)])

        remaining_V1 = V1[:, 1:]
        remaining_V2 = V2[:, np.logical_not(near_zero)]
        for i in range(remaining_V2.shape[1]):
            assert (np.allclose(remaining_V1[:, i],  remaining_V2[:, i]) or
                    np.allclose(remaining_V1[:, i], -remaining_V2[:, i]))


class Test_speigh_1(object):
    def test_1(self):
        # test with indefinite A matrix, identity B
        n = 4
        A = rand_sym(n)
        B = np.eye(n)

        w0, v0 = speigh(A, B, rho=0)
        w, V = eigh(A, B)
        np.testing.assert_array_almost_equal(w[0], w0)
        np.testing.assert_array_almost_equal(v0**2, V[:,0]**2)

    def test_2(self):
        # test with indefinite B matrix, indefinite B
        n = 4
        A = rand_sym(n)
        B = rand_pos_semidef(n)

        w0, v0 = speigh(A, B, rho=0)
        w, V = eigh(A, B)
        np.testing.assert_array_almost_equal(w[0], w0)
        np.testing.assert_array_almost_equal(v0**2, V[:,0]**2)

    def test_3(self):
        # test with positive semidefinite A matrix, and diagonal
        # matrix B
        n = 4
        A = rand_pos_semidef(n)
        B = np.diag(np.random.randn(n)**2)

        w0, v0 = speigh(A, B, rho=0)
        w, V = eigh(A, B)
        np.testing.assert_array_almost_equal(w[0], w0)
        np.testing.assert_array_almost_equal(v0**2, V[:,0]**2)

    def test_4(self):
        # test with positive semidefinite A matrix, and general
        # matrix B
        n = 4
        A = rand_pos_semidef(n)
        B = rand_pos_semidef(n) + np.eye(n)

        w0, v0 = speigh(A, B, rho=0)
        w, V = eigh(A, B)
        np.testing.assert_array_almost_equal(w[0], w0)
        np.testing.assert_array_almost_equal(v0**2, V[:,0]**2)


class Test_speigh_2(object):

    def test_1(self):
        # test with indefinite A matrix, identity B
        n = 4
        x =    np.array([1.0, 2.0, 3.0, 0.0001])
        x = x / np.sqrt(np.sum(x**2))

        A = np.outer(x, x)
        B = np.eye(n)
        w, V = eigh(A, B)

        w0, v0 = speigh(A, B, rho=0.01)

        x_sp = np.array([1.0, 2.0, 3.0, 0])
        x_sp = x_sp / np.sqrt(np.sum(x_sp**2))
        np.testing.assert_array_almost_equal(v0, x_sp)

    def test_2(self):
        n = 4
        # build matrix with specified first generalized eigenvector
        A, B = build_lowrank(n, [1, 2, 0.001, 3], seed=0)
        w, V = eigh(A, B)

        v1 = speigh(A, B, method=1, rho=1e-4)[1][2]
        v2 = speigh(A, B, method=2, rho=1e-4)[1][2]
        v3 = speigh(A, B, method=3, rho=1e-4)[1][2]
        v4 = speigh(A, B, method=4, rho=1e-4)[1][2]

        np.testing.assert_almost_equal(v1, 0.001)
        np.testing.assert_almost_equal(v2, 0)
        np.testing.assert_almost_equal(v3, 0)
        np.testing.assert_almost_equal(v4, 0)

    def test_3(self):
        n = 10
        A = rand_sym(n, seed=1)
        B = rand_pos_semidef(n, seed=1)
        w1, V1 = speigh(A, B, method=1, rho=10)
        w2, V2 = speigh(A, B, method=2, rho=10)
        w3, V3 = speigh(A, B, method=3, rho=10)

        np.testing.assert_almost_equal(w2, w3)
        np.testing.assert_almost_equal(V2, V3)

# cpdef double solve_admm2(const double[::1] b, const double[::1] w,
#                         const double[:, ::1] B, double[::1] x,
#                         double B_maxeig, double[:, ::1] A,
#                         double tol=1e-4, int maxiter=100, int verbose=0,
#                         method="ista"):

class TestSolvers(object):
    def test_admm2_vs_cvxpy(self):
        b = np.array([-50.852, -66.227,  -8.96 , -86.548, -24.175])
        w = np.array([  6.104e-01,   1.070e+01,   1.250e+01,   8.469e-01,   1.251e+02])
        B = np.array([[ 4.805,  0.651,  0.611, -4.98 , -1.448],
                      [ 0.651,  6.132, -1.809,  0.613,  4.838],
                      [ 0.611, -1.809,  4.498,  0.055, -4.548],
                      [-4.98 ,  0.613,  0.055,  9.841,  2.17 ],
                      [-1.448,  4.838, -4.548,  2.17 ,  9.949]])

        x1, f1 = solve_cvxpy(b, w, B)

        x2 = np.zeros(len(b))
        f2 = solve_admm2(b, w, B, x2, np.max(scipy.linalg.eigvals(B)), np.ascontiguousarray(scipy.linalg.cholesky(B)))

        np.testing.assert_almost_equal(f1, f2, decimal=2)
        np.testing.assert_array_almost_equal(x1, x2, decimal=4)

