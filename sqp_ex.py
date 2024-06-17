import numpy as np
import quadprog

def hildreth_qp(E, F, M, gamma, max_iter=1000, tol=1e-6):
    # Compute H and K matrices
    E_inv = np.linalg.pinv(E)
    H = M @ E_inv @ M.T
    K = gamma + M @ E_inv @ F

    # Initialize
    n = gamma.shape[0]

    lamb = np.zeros(n).reshape(-1, 1)
    w = np.zeros(n).reshape(-1, 1)

    # Iterate to update lambda
    for m in range(max_iter):
        lamb_old = lamb.copy()

        for i in range(n):
            sum1 = 0
            sum2 = 0

            for j in range(i):
                sum1 += H[i, j] * lamb[j]
            for j in range(i + 1, n):
                sum2 += H[i, j] * lamb_old[j]
            w[i] = (-sum1 - sum2 - K[i]) / H[i, i]
            lamb[i] = max(w[i], 0)

        # Check for convergence
        if np.linalg.norm(lamb - lamb_old) < tol:
            break

    # Compute the solution x
    x = -E_inv @ (F + M.T @ lamb)

    return x

def solve_qp_quadprog(E, F, M, gamma):
    G = np.linalg.cholesky(E).T
    a = -F
    C = -M.T
    b = -gamma

    meq = 0  # Number of equality constraints
    sol = quadprog.solve_qp(G, a, C, b, meq)
    return sol[0]



# Example usage
E = np.array([[2, 0], [0, 2]], dtype=np.double)
F = np.array([-2, -5], dtype=np.double)
M = np.array([[1, 2], [1, -4], [-1, -2]], dtype=np.double)
gamma = np.array([3, 2, -1], dtype=np.double)

x = solve_qp_quadprog(E, F, M, gamma)
y = hildreth_qp(E, F, M, gamma)

print("Solution from hildreth_qp:", y)
print("Solution from solve_qp_quadprog:", x)
