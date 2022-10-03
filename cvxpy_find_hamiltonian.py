#!/usr/bin/env python3
import cvxpy as cp
import numpy as np
import sys
from cov_matrix import *
import scipy.linalg
import math
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

# Find the Q from the covariance matrix and test whether it is correct
def generate_problem(filepath_cov_matrix):
    cov_matrix = np.loadtxt(filepath_cov_matrix)
    eigenvalues, eigenvectors = ordered_eigenspectrum(1j * cov_matrix)
    Q = transformation_matrix(cov_matrix)
    test = Q @ cov_matrix @ Q.T
    print(test)
    return eigenvalues, Q


# Define and solve the CVXPY problem.
def solve(filepath_cov_matrix):
    # find number of eigenvalues (size of cov matrix) and Q
    eigenvalues, Q = generate_problem(filepath_cov_matrix)
    n = len(eigenvalues)
    print(n)
    print("the eigenvalues are: {}".format(eigenvalues))

    # half the eigenvalues are positive, which is what Q finds:
    n_pos_eig = round(n / 2)

    # set the variable and create the CVXPY problem
    x = cp.Variable(n_pos_eig)
    diag = [np.array([[0, -1], [1, 0]]) * x[i] for i in range(0, n_pos_eig)]
    s = np.zeros((2, 2))
    l = np.array([s for i in range(0, n_pos_eig)])

    S = cp.hstack([diag[1], np.zeros((2, 2 * (n_pos_eig - 1)))])
    for i in range(1, n_pos_eig - 1):
        L = np.zeros((2, 2 * i))
        R = np.zeros((2, 2 * (n_pos_eig - i - 1)))
        S_temp = cp.hstack([L, diag[i], R])
        S = cp.vstack([S, S_temp])
    S_last = cp.hstack([np.zeros((2, 2 * (n_pos_eig - 1))), diag[n_pos_eig - 1]])
    S = cp.vstack([S, S_last])
    print("The shape of S is {} and the shape of Q is {}.".format(S.shape, Q.shape))
    A = -Q.T @ S @ Q
    A_prob = [cp.abs(A[i, j]) * i * j for i in range(n) for j in range(i + 2, n)]

    # solve the CVXPY problem:
    prob = cp.Problem(
        cp.Minimize(cp.sum(A_prob)),
        [x >= 0] + [x[3] == 1],
    )
    prob.solve(
        verbose=True,
        max_iter=50000,
        eps_abs=0.001,
    )

    # return the optimal value for the sum of off-diagonal elements as well as
    # the solution
    print("x")
    print(Q @ A.value @ Q.T)
    print("y")
    print(x.value)
    return prob.value, x.value


if __name__ == "__main__":
    opt_value, solution = solve("path_to_cmatrix")

    print("\nThe optimal value is {}".format(opt_value))
    print("A solution is {} (ignoring the dual solution)".format(solution))
