"""Runnable SciPy BLAS/LAPACK examples.

Requires NumPy and SciPy.  The low-level wrappers return arrays (and an
``info`` status for LAPACK), so every result is kept and checked explicitly.
"""

import numpy as np
from scipy.linalg import blas, lapack


def require_info(name, info):
    """Raise when a LAPACK routine reports an invalid argument or failure."""
    if info != 0:
        raise RuntimeError(f"{name} failed with info={info}")


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def main():
    np.set_printoptions(precision=5, suppress=True)

    # A is nonsingular; T is an upper-triangular matrix for triangular BLAS.
    A = np.array(
        [[4.0, 1.0, 2.0],
         [0.0, 3.0, -1.0],
         [2.0, 0.0, 5.0]]
    )
    B = np.array(
        [[1.0, 0.0, 2.0],
         [0.0, 1.0, 1.0],
         [2.0, 1.0, 0.0]]
    )
    T = np.triu(A)
    n = A.shape[0]
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    # BLAS Level 1: vector-vector operations.
    dot_xy = blas.ddot(x, y)
    y_axpy = blas.daxpy(x, y.copy(), a=2.0)  # y + 2*x
    y_copy = blas.dcopy(x, y=np.empty_like(x))
    x_swap, y_swap = blas.dswap(x.copy(), y.copy())
    norm_x = blas.dnrm2(x)

    # BLAS Level 2: matrix-vector and rank-1 operations.
    y_mv = blas.dgemv(1.0, A, x, beta=0.5, y=y.copy())
    x_tri = blas.dtrmv(T, x.copy(), lower=False, trans=0, diag=0)
    A0 = np.eye(n)
    A_rank1 = blas.dger(1.0, x, y, a=A0.copy())  # A0 + x*y.T

    # BLAS Level 3: matrix-matrix and symmetric rank-k operations.
    C = blas.dgemm(1.0, A, B)  # beta defaults to zero
    D = blas.dtrmm(1.0, T, np.eye(n), side=0, lower=False, trans_a=0, diag=0)
    syrk_upper = blas.dsyrk(1.0, A, lower=False)  # upper triangle of A*A.T
    syrk = np.triu(syrk_upper) + np.triu(syrk_upper, 1).T

    assert_close(dot_xy, x @ y)
    assert_close(y_axpy, y + 2.0 * x)
    assert_close(y_copy, x)
    assert_close(x_swap, y)
    assert_close(y_swap, x)
    assert_close(norm_x, np.linalg.norm(x))
    assert_close(y_mv, A @ x + 0.5 * y)
    assert_close(x_tri, T @ x)
    assert_close(A_rank1, A0 + np.outer(x, y))
    assert_close(C, A @ B)
    assert_close(D, T)
    assert_close(syrk, A @ A.T)

    # LAPACK LU factorization and solve.  Keep pivots from the same factor.
    lu, piv, info = lapack.dgetrf(A.copy())
    require_info("dgetrf", info)
    b = np.array([[1.0], [2.0], [3.0]])
    solution_lu, info = lapack.dgetrs(lu, piv, b.copy())
    require_info("dgetrs", info)
    assert_close(A @ solution_lu, b)

    # LAPACK Cholesky factorization and solve of an explicitly SPD matrix.
    M = np.array(
        [[1.0, 2.0, 0.0],
         [0.0, 1.0, 1.0],
         [2.0, 0.0, 1.0]]
    )
    A_spd = M.T @ M + 0.5 * np.eye(n)
    chol, info = lapack.dpotrf(A_spd.copy(), lower=True)
    require_info("dpotrf", info)
    solution_chol, info = lapack.dpotrs(chol, b.copy(), lower=True)
    require_info("dpotrs", info)
    assert_close(A_spd @ solution_chol, b)

    # LAPACK QR factorization.  dgeqrf returns four values.
    qr, tau, _, info = lapack.dgeqrf(A.copy())
    require_info("dgeqrf", info)
    Q, _, info = lapack.dorgqr(qr, tau)
    require_info("dorgqr", info)
    R = np.triu(qr)
    assert_close(Q @ R, A)

    # LAPACK least squares.  dgels returns (factorized_a, overwritten_b, info),
    # not the five-value output of a high-level least-squares routine.
    A_ls = np.array(
        [[1.0, 0.0],
         [1.0, 1.0],
         [1.0, 2.0],
         [1.0, 3.0]]
    )
    b_ls = np.array([[1.0], [2.0], [2.0], [4.0]])
    _, rhs_ls, info = lapack.dgels(A_ls.copy(), b_ls.copy(), trans="N")
    require_info("dgels", info)
    solution_ls = rhs_ls[:A_ls.shape[1], :]
    residual_norm = np.linalg.norm(A_ls @ solution_ls - b_ls)

    print("BLAS dot              =", dot_xy)
    print("BLAS AXPY             =", y_axpy)
    print("BLAS gemv             =", y_mv)
    print("BLAS gemm shape       =", C.shape)
    print("LU solution           =", solution_lu.ravel())
    print("Cholesky solution     =", solution_chol.ravel())
    print("QR reconstruction err =", np.linalg.norm(Q @ R - A))
    print("least-squares solution=", solution_ls.ravel())
    print("least-squares residual=", residual_norm)
    print("All BLAS/LAPACK checks passed.")


if __name__ == "__main__":
    main()
