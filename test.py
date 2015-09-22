import numpy as np
import scipy.linalg
from msmbuilder.decomposition._speigh import speigh, solve_admm, scdeflate


def soft_thresh(k, a):
    return np.maximum(a-k, 0) + np.minimum(a+k, 0)


def project(v, B):
    norm2 = np.dot(v, B).dot(v)
    if norm2 <= 1:
        return v
    return v / np.sqrt(norm2)


def solve_admm2_x_cvxpy(b, A, v, w, rho):
     # 1/2 ||x-b||^2  +  (p/2) ||Ax - v||^2  +  ||D(w)x||_1
     import cvxpy as cp
     n = len(b)
     x = cp.Variable(n)
     objective = cp.Minimize(
         (1/2) * cp.norm2(x - b)**2  \
         + rho/2 * cp.norm2(A*x - v)**2 \
         + cp.norm1(cp.diag(w)*x))

     problem = cp.Problem(objective)
     problem.solve(solver='SCS')
     x = np.asarray(x.value)[:,0]

     f = 0.5*np.linalg.norm(x-b)**2 + 0.5*rho*np.linalg.norm(A.dot(x)-v)**2 + np.sum(np.abs(np.multiply(w, x)))
     print('solve_admm2_x_cvxpy f', f, x)

     return x



def solve_admm2_x_fista(b, A, v, w, rho, x0):
    # 1/2 ||x-b||^2  +  (p/2) ||Ax - v||^2  +  ||D(w)x||_1

    n = len(b)
    I = np.eye(n)
    y = x0.copy()
    x = np.zeros(n)
    x_old = np.zeros(n)

    #import scipy.linalg
    #step = 1/(rho*np.max(scipy.linalg.eigvalsh(A.T.dot(A))))
    # print(step)
    # print(step)
    step = 1e-5

    term1 = rho*A.T.dot(A) + np.eye(n)
    term2 = rho*A.T.dot(v) + b

    t = 1
    eps = np.finfo(np.float).eps

    f, old_f = np.finfo(np.float).max, np.finfo(np.float).max
    for i in range(1000):
        old_f = f
        x_old[:] = x[:]

        # x_{t+1} = \argmin
        #  ||D(w)x||_1 + (1/2n) ||x - x_t*n\grad(f)||^2

        x = soft_thresh(np.abs(w)*step, y - step* (term1.dot(y)-term2))

        t_new = (1 + np.sqrt(1 + 4*t**2)) / 2
        y = x + (t-1)/t_new * (x - x_old)
        t = t_new

        f = 0.5*np.linalg.norm(y-b)**2 + 0.5*rho*np.linalg.norm(A.dot(y)-v)**2 + np.sum(np.abs(np.multiply(w, y)))
        if i > 1 and abs(f - old_f) < eps:
            break

    print('solve_admm2_x_fista f', f, i)
    return y
        # if abs(f - old_f) < eps:
        #     print('f', i)
        #     if f < old_f:
        #         return y
        #     return x_old


def solve_admm2_x_ista(b, A, v, w, rho, x0):
    # 1/2 ||x-b||^2  +  (p/2) ||Ax - v||^2  +  ||D(w)x||_1

    n = len(b)
    I = np.eye(n)
    x = x0.copy()

    import scipy.linalg
    step = 1/(10*np.max(scipy.linalg.eigvalsh(A.T.dot(A))))

    term1 = rho*A.T.dot(A) + np.eye(n)
    term2 = rho*A.T.dot(v) + b

    eps = np.finfo(np.float).eps

    f, old_f = np.finfo(np.float).max, np.finfo(np.float).max
    for i in range(1000):
        old_f = f
        # x_{t+1} = \argmin
        #  ||D(w)x||_1 + (1/2n) ||x - x_t*n\grad(f)||^2

        grad = term1.dot(x) - term2
        x = soft_thresh(np.abs(w)*step, x - step*grad)
        f = 0.5*np.linalg.norm(x-b)**2 + 0.5*rho*np.linalg.norm(A.dot(x)-v)**2 + np.sum(np.abs(np.multiply(w, x)))

        if i > 1 and abs(f - old_f) < eps or f > old_f:
            break

    print('solve_admm2_x_ista f', f, i)
    return x




def solve_admm2_x_admm(b, A, v, w, rho, x0):
    # 1/2 ||x-b||^2  +  (p/2) ||Ax - v||^2  +  ||D(w)x||_1

    "min_x (1/2) ||x-b||^2  +  (p/2) ||Ax - v||^2  + (gamma/2) ||x-z||^2"
    "min_x 1/2 x.T ((1+gamma) I + p A.T A) x  -  x.T (b + gamma*z + p A.Tv)"
    "min_x 1/2 x.T P x - q.T x"
    """
    where P = ((1+gamma) I + p A.T A
          q = b + gamma*z + p A.Tv
    """

    n = len(b)
    gamma = 16
    x = np.copy(x0)
    z = np.copy(x0)
    z_old = x0.copy()
    u = np.zeros(n)

    for i in range(200):
        z_old[:] = z[:]
        # x update
        # min_x (1/2) ||x-b||^2 + (p/2) ||Ax - v||^2  + (gamma/2) ||x-(z+u)||^2
        P = (1+gamma)*np.eye(n) + rho * A.T.dot(A)
        q = b + gamma*(z-u) + rho*A.T.dot(v)
        x = scipy.linalg.solve(P, q)

        # import cvxpy as cp
        # x2 = cp.Variable(n)
        # cp.Problem(cp.Minimize(
        #     (1/2)     *  cp.norm2(x2-b)**2 + \
        #     (rho/2)   *  cp.norm2(A*x2-v)**2 + \
        #     (gamma/2) *  cp.norm2(x2-(z+u))**2)).solve()
        # np.testing.assert_array_almost_equal(x, np.asarray(x2.value)[:,0], decimal=3)

        # z update
        # min_z ||D(w)z||_1 + gamma/2 ||z - (x+u)||^2

        z = soft_thresh(np.abs(w)/gamma, (x+u))
        # import cvxpy as cp
        # z2 = cp.Variable(n)
        # cp.Problem(cp.Minimize(
        #     cp.norm1(cp.diag(w)*z2) + (gamma/2)*cp.norm2(z2-(x+u))**2
        # )).solve()
        # np.testing.assert_array_almost_equal(z, np.asarray(z2.value)[:,0], decimal=3)

        # u update
        r = x - z
        u = u + r

        r_norm = np.linalg.norm(r)                  # primal residual
        s_norm = gamma * np.linalg.norm(z - z_old)    # dual residual

        if r_norm < np.sqrt(n)*5e-4 and s_norm < np.sqrt(n)*5e-4:
            break

        # print(r_norm*np.sqrt(n), s_norm*np.sqrt(n))
        # print(0.5*np.dot(x-b, x-b) + rho/2*np.dot(np.dot(A,x)-v, np.dot(A,x)-v) + np.sum(np.abs(np.multiply(w, x))))
        #print(0.5*np.dot(z-b, z-b) + rho/2*np.dot(np.dot(A,z)-v, np.dot(A,z)-v) + np.sum(np.abs(np.multiply(w, z))))
        #print()


        # print('residuals ', r_norm, s_norm)
    # print('  i', i)
    # print('x', 0.5*np.dot(x-b, x-b) + rho/2*np.dot(np.dot(A,x)-v, np.dot(A,x)-v) + np.sum(np.abs(np.multiply(w, x))))
    # print('z', 0.5*np.dot(z-b, z-b) + rho/2*np.dot(np.dot(A,z)-v, np.dot(A,z)-v) + np.sum(np.abs(np.multiply(w, z))))
    # print()

    f = 0.5*np.linalg.norm(z-b)**2 + 0.5*rho*np.linalg.norm(A.dot(x)-z)**2 + np.sum(np.abs(np.multiply(w, z)))
    print('solve_admm2_x_admm f', f, z)

    return z


    # f1 = 0.5*np.dot(x-b, x-b) + rho/2*np.dot(np.dot(A,x)-v, np.dot(A,x)-v) + gamma/2*np.dot(x-z, x-z)
    #
    # f2 = 0.5 * np.dot(x, ((1+gamma)*np.eye(n) + rho * A.T.dot(A))).dot(x) \
    #      -  x.dot(b + gamma*z + rho*A.T.dot(v))  \
    #      + (0.5 * np.dot(b, b) + rho/2*np.dot(v,v) + gamma/2*np.dot(z,z))
    #
    # print(f1, f2)



def solve_admm2(b, w, B, x0=None, tol=1e-6, maxiter=100):
    """Solve a convex optimization problem with ADMM

    Minimize    1/2 ||x-b||^2 + ||D(w)x||_1
    subject to  x^T B x <= 1

    Parameters
    ----------
    b : array, shape=(n,)
    w : array, shape=(n,)
    B : array, shape=(n,n)
    x0 : array, shape=(n,) default=None
    tol : float, default=1e-6
    maxiter : int, default=100
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    x = np.copy(x0)
    z = np.copy(x0)
    z_old = np.zeros(n)
    u = np.zeros(n)
    rho = 16
    I = np.eye(n)

    A = scipy.linalg.cholesky(B)
    tol = 1e-4

    for i in range(maxiter):
        z_old[:] = z[:]

        #x0 = solve_admm2_x_admm(b, A, v=(z-u), w=w, rho=rho, x0=x)
        x1 = solve_admm2_x_fista(b, A, v=(z-u), w=w, rho=rho, x0=x)
        #x1 = solve_admm2_x_ista(b, A, v=(z-u), w=w, rho=rho, x0=x)
        #x1 = solve_admm2_x_ista(b, A, v=(z-u), w=w, rho=rho, x0=x)
        #print()
        #x2 = solve_admm2_x_cvxpy(b, A, v=(z-u), w=w, rho=rho)

        x = x1

        z = project(A.dot(x) + u, I)
        u = u + (A.dot(x) - z)

        r = np.linalg.norm(A.dot(x)-z)         # primal residual
        s = rho * np.linalg.norm(z - z_old)    # dual residual

        # print(np.dot(x-b, x-b) + np.sum(np.abs(np.multiply(w, x))))
        # print('residuals ', r, s, 'rho', rho)
        if r < np.sqrt(n)*tol and s < np.sqrt(n)*tol:
            break
        # fun1 =  0.5*np.dot(x-b, x-b) + np.sum(np.abs(np.multiply(w, x)))
        # fun2 =  0.5*np.dot(z-b, z-b) + np.sum(np.abs(np.multiply(w, z)))
        # print('admm2 fun', fun1, np.dot(x,B).dot(x))

        # Varying the penalty parameter, eq. (3.13)
        if r > 10 * s and rho < 2**10:
            rho *= 2
            u /= 2
        elif s > 10 * s and rho > 2**-10:
            rho /= 2
            u *= 2

    x = project(x, B)
    print('i', i)
    # print(np.dot(x,B).dot(x))
    fun =  0.5*np.dot(x-b, x-b) + np.sum(np.abs(np.multiply(w, x)))

    return x, fun


def solve_admm_slow(b, w, B, lambd, x0=None, tol=1e-6, maxiter=100):
    """Solve a convex optimization problem with ADMM

    Minimize    1/2 ||x-b||^2 + lambd ||D(w)x||_1
    subject to  x^T B x <= 1

    Parameters
    ----------
    b : array, shape=(n,)
    w : array, shape=(n,)
    B : array, shape=(n,n)
    x0 : array, shape=(n,) default=None
    tol : float, default=1e-6
    maxiter : int, default=100
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    x = np.copy(x0)
    z = np.copy(x0)
    z_old = np.zeros(n)
    u = np.zeros(n)
    rho = 16

    for i in range(maxiter):
        z_old = z

        x = (1/(rho+1)) * soft_thresh(np.abs(lambd*w), b + rho*(z - u))
        z = project(x + u, B)
        u = u + (x - z)

        r = np.linalg.norm(x-z)                # primal residual
        s = rho * np.linalg.norm(z - z_old)    # dual residual

        # print('residuals ', r, s, 'rho', rho)
        if r < np.sqrt(n)*tol and s < np.sqrt(n)*tol:
            break

        # Varying the penalty parameter, eq. (3.13)
        if r > 10 * s and rho < 2**10:
            rho *= 2
            u /= 2
        elif s > 10 * r and rho > 2**-10:
            rho /= 2
            u *= 2

    fun =  0.5*np.dot(x-b, x-b) + lambd*np.sum(np.abs(np.multiply(w, x)))
    return x, fun


def speigh_slow(A, B, rho, eps=1e-6, tol=1e-8, maxiter=100, verbose=False):
    N = len(A)
    rho_e = rho / scipy.special.log1p(1/eps)
    tau = 0.1 + max(0, -np.min(scipy.linalg.eigvalsh(A)))
    x = scipy.linalg.eigh(A, B, eigvals=(N-1, N-1))[1][:,0]

    fun, old_fun = np.inf, np.inf
    for i in range(maxiter):
        old_fun = fun
        fun = np.dot(x, A).dot(x) - rho_e * np.sum(np.log(np.abs(x) + eps))
        if verbose:
            print('f,x', fun, x)
        if abs(old_fun - fun) < tol:
            break

        w = rho_e / (2*tau*(np.abs(x) + eps))
        b = np.dot(A, x)/tau + x
        x, f2 = solve_cvxpy(b, w, B)
        if fun > 8:
            print(repr(b), repr(w), repr(B))
            raise ValueError()

        if verbose:
            print('  f2', f2)

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
        v[mask] = 1.0 / np.sqrt(Bk[0,0])
        u = Ak[0,0] / Bk[0,0]
    else:
        gevals, gevecs = scipy.linalg.eigh(
            Ak, Bk, eigvals=(Ak.shape[0]-1, Ak.shape[0]-1))
        # Usually slower to use sparse linear algebra here
        # gevals, gevecs = scipy.sparse.linalg.eigsh(
        #     A=Ak, M=Bk, k=1, v0=x[mask], which='LA')
        u = gevals[0]
        v = np.zeros(N)
        v[mask] = gevecs[:, 0]
        v *= np.sign(np.sum(v))

    return u, v

def solve_cvxpy(b, w, B):
    """Solve a convex optimization problem using cvxpy

    Minimize    1/2 ||x-b||^2 + ||D(w)x||_1
    subject to  x^T B x <= 1

    Parameters
    ----------
    b : array, shape=(n,)
    w : array, shape=(n,)
    B : array, shape=(n,n)

    Returns
    -------
    xf : array, shape=(n,)
        The optimal value of x
    fun : float
        The value of the objective function at ``xf``
    """
    import cvxpy as cp
    n = len(b)
    x = cp.Variable(n)
    #objective = cp.Minimize(
    #    0.5 * cp.norm2(x-b)**2 + lambd*cp.norm1(cp.diag(w) * x))

    # more accurate this way, expanding out the norm
    objective = cp.Minimize(
        0.5*cp.norm2(x)**2 - x.T*b + cp.norm1(cp.diag(w) * x))

    constraints = [cp.quad_form(x, B) <= 1]
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver='SCS')
    if problem.status != 'optimal':
        print(problem, problem.status)
        raise ValueError('No solution found')

    x = np.asarray(x.value)[:,0]
    x[np.abs(x) < 1e-10] = 0
    fun = 0.5*np.dot(x-b, x-b) + np.sum(np.abs(np.multiply(w, x)))
    return x, fun

################################################################################
################################################################################
################################################################################

def test_1():
    random = np.random.RandomState(0)
    wish = scipy.stats.wishart(scale=np.eye(5), seed=random)
    A = wish.rvs()
    B = wish.rvs() + np.eye(5)

    print(speigh_slow(A, B, rho=1, tol=1e-8, verbose=1))
    print()
    print(speigh(A, B, rho=1, verbose=1))


def test_admm_vs_cvxpy_1():
    # random positive definite matrix
    random = np.random.RandomState(2)
    wish = scipy.stats.wishart(scale=np.eye(3), seed=random)
    for B in [np.eye(3), wish.rvs()]:
        b = random.randn(3)
        w = random.randn(3)
        lambd = 2

        x0 = np.zeros(3)
        solve_admm(b=b, w=lambd*w, B=B, x=x0)
        f0 = 0.5*np.dot(x0-b, x0-b) + lambd*np.sum(np.abs(np.multiply(w, x0)))

        x1, f1 = solve_cvxpy(b=b, w=w, B=B, lambd=lambd)
        # not sure what tolerance is appropriate for these tests
        np.testing.assert_almost_equal(f0, f1, decimal=3)
        np.testing.assert_array_almost_equal(x0, x1, decimal=3)


def test_admm_vs_cvxpy_2():
    n = 1000
    random = np.random.RandomState(0)
    b = 10*random.randn(n)
    w = 10*random.randn(n)
    B = scipy.stats.wishart(scale=np.eye(n), seed=random).rvs()
    # B = np.eye(n)

    #x1, f1 = solve_cvxpy(b, w, B)
    #print('cvxpy  ', f1, np.sum(np.abs(x1)>1e-6))

    import time
    start = time.time()

    x2 = np.zeros(n)
    f2 = solve_admm(b, w, B, x2, maxiter=1000)
    print('admm 1 ', f2, np.sum(np.abs(x2)>1e-6))
    t1 = time.time()
    print('  t1', t1 - start)

    x3, f3 = solve_admm2(b, w, B, maxiter=1000)
    print('admm 2 ', f3, np.sum(np.abs(x3)>1e-6))
    print('  t2', time.time() - t1)


def test_admm_vs_cvxpy_3():
    b = np.array([-50.852, -66.227,  -8.96 , -86.548, -24.175])
    w = np.array([  6.104e-01,   1.070e+01,   1.250e+01,   8.469e-01,   1.251e+02])
    B = np.array([[ 4.805,  0.651,  0.611, -4.98 , -1.448],
                  [ 0.651,  6.132, -1.809,  0.613,  4.838],
                  [ 0.611, -1.809,  4.498,  0.055, -4.548],
                  [-4.98 ,  0.613,  0.055,  9.841,  2.17 ],
                  [-1.448,  4.838, -4.548,  2.17 ,  9.949]])

    x1, f1 = solve_cvxpy(b, w, B)
    print('cvxpy  ', f1, x1)

    x2, f2 = solve_admm_slow(b, w, B, lambd=1)
    print('admm 1 ', f2, x2)

    x3, f3 = solve_admm2(b, w, B)
    print('admm 2 ', f3, x3)


def test_admm3():
    from mdtraj.utils import timing
    from msmbuilder.decomposition._speigh import solve_admm, solve_admm2


    n = 1000
    random = np.random.RandomState(0)
    b = 100*random.randn(n)
    w = 100*random.randn(n)
    B = scipy.stats.wishart(scale=np.eye(n), seed=random).rvs()


    x2 = np.zeros(n)
    with timing('admm1'):
        f2 = solve_admm(b, w, B, x2, maxiter=1000)
    print('admm 1 ', f2, np.sum(np.abs(x2)>1e-6))


    x3 = np.zeros(n)
    B_maxeig = np.max(scipy.linalg.eigvalsh(B))
    B_chol = np.ascontiguousarray(scipy.linalg.cholesky(B))

    with timing('admm2'):
        f3 = solve_admm2(b, w, B, x3, B_maxeig, B_chol, tol=1e-3, maxiter=1000, verbose=2)
    print('admm 2 ', f3, np.sum(np.abs(x3)>1e-6))



def test_admm4():
    n = 100
    random = np.random#.RandomState(0)
    A = scipy.stats.wishart(scale=np.eye(n), seed=random).rvs()
    B = scipy.stats.wishart(scale=np.eye(n), seed=random).rvs() #+ np.eye(n)
    B = np.eye(n)
    B[0,1] = 0.1
    B[1,0] = 0.1
    B[10,30] = 0.5
    B[30,10] = 0.5

    print(scipy.linalg.eigvalsh(A,B))
    print(np.linalg.eigvalsh(B))

    from msmbuilder.decomposition._speigh import speigh

    print(speigh(A, B, rho=1e-1, method=1))
    # print(speigh(A, B, rho=1e-1, method=3))


def build_dataset():
    from msmbuilder.example_datasets import DoubleWell
    slow = [DoubleWell(random_state=0).get()['trajectories'][0][::10]]
    data = []

    # each trajectory is a double-well along the first dof,
    # and then 9 degrees of freedom of gaussian white noise.
    for s in slow:
        t = np.hstack((s, np.random.randn(len(s), 500)))
        data.append(t)
    return data

def test_sparsetica_1():
    from mdtraj.utils import timing
    from msmbuilder.decomposition import tICA
    ds = build_dataset()
    tica = tICA().fit(ds)

    A = tica.offset_correlation_
    B = tica.covariance_
    gamma = 0.06
    B = B + (gamma / len(B)) * np.trace(B)+np.eye(len(B))

    with timing():
        print(speigh(A, B, rho=1e-2, method=1, verbose=1, tol=1e-8))
    #print(ds)







