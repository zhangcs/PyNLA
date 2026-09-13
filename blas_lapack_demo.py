"""Short NumPy/SciPy BLAS and LAPACK demo."""

import numpy as np
from scipy.linalg import blas, lapack


np.set_printoptions(precision=4, suppress=True)

A = np.array([[3.0, 1.0, 2.0],
              [1.0, 4.0, 0.0],
              [0.0, 2.0, 5.0]])
B = np.array([[1.0, 0.0, 2.0],
              [0.0, 1.0, 1.0],
              [2.0, 1.0, 0.0]])
b = np.array([1.0, 2.0, 3.0])
n = A.shape[0]

# NumPy high-level linear algebra (LAPACK-backed, not a BLAS level)
x_solve = np.linalg.solve(A, b)
A_inv = np.linalg.inv(A)  # For demonstration; prefer solve(A, b) for a system.
print("solve residual :", np.linalg.norm(A @ x_solve - b))
print("inverse residual:", np.linalg.norm(A @ A_inv - np.eye(n)))

# BLAS Level 1: vector-vector operations
u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])
v_axpy = blas.daxpy(u, v.copy(), a=2.0)       # v + 2u
u_copy = blas.dcopy(u, y=np.empty_like(u))    # explicit destination

# BLAS Level 2: matrix-vector and rank-1 operations
Au = blas.dgemv(1.0, A, u)                    # Au
A0 = np.eye(n)
A_rank1 = blas.dger(1.0, u, v, a=A0.copy())     # A0 + uv^T

# BLAS Level 3: matrix-matrix operation
AB = blas.dgemm(1.0, A, B)                    # AB

assert np.allclose(v_axpy, v + 2.0 * u)
assert np.allclose(u_copy, u)
assert np.allclose(Au, A @ u)
assert np.allclose(A_rank1, A0 + np.outer(u, v))
assert np.allclose(AB, A @ B)
print("BLAS checks     : OK")

# LAPACK: Cholesky factorization of an explicitly SPD matrix
M = np.array([[1.0, 2.0, 0.0],
              [0.0, 1.0, 1.0],
              [2.0, 0.0, 1.0]])
A_spd = M.T @ M + 0.5 * np.eye(n)
chol, info_chol = lapack.dpotrf(A_spd.copy(), lower=True)
L = np.tril(chol)

# LAPACK: QR factorization; dgeqrf returns four values
qr, tau, _, info_qr = lapack.dgeqrf(A.copy())
Q, _, info_orgqr = lapack.dorgqr(qr, tau)
R = np.triu(qr)

# LAPACK: overdetermined least-squares problem; dgels returns three values
A_ls = np.array([[1.0, 0.0],
                 [1.0, 1.0],
                 [1.0, 2.0],
                 [1.0, 3.0]])
b_ls = np.array([[1.0], [2.0], [2.0], [4.0]])
_, rhs_ls, info_ls = lapack.dgels(A_ls.copy(), b_ls.copy())
x_ls = rhs_ls[:A_ls.shape[1], 0]

assert info_chol == info_qr == info_orgqr == info_ls == 0
print("Cholesky error  :", np.linalg.norm(L @ L.T - A_spd))
print("QR error        :", np.linalg.norm(Q @ R - A))
print("least-squares x :", x_ls)
