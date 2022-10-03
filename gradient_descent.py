#!/usr/bin/env python
import numpy as np
import scipy
import scipy.stats
from cov_matrix import *


def squared_error(target_cov_matrix, hamiltonian):
    """
    calculate the covariance matrix from a hamiltonian, compare it to the
    target covariance matrix, and return the squared error.
    """
    cov_matrix = groundstate_correlation_matrix(hamiltonian)
    return np.sum((cov_matrix - target_cov_matrix) ** 2)


def perturb_matrix_at(M, i, j, sigma):
    """
    Add a random pertubation with standard deviation sigma onto the (i,j)th
    matrix element.
    """
    pertubation = np.random.normal(scale=sigma)
    M[i, j] += pertubation
    return pertubation, M


def generate_perturbed_square_errors(target_cov_matrix, hamiltonian, i, j, sigma):
    """
    a generator over the above function that also calculates the squared errors
    for each pertubation.
    """
    while True:
        pertubation, perturbed = perturb_matrix_at(hamiltonian.copy(), i, j, sigma)
        yield pertubation, squared_error(target_cov_matrix, hamiltonian)


def stochastic_derivative(target_cov_matrix, hamiltonian, i, j, sigma, n):
    """
    generte n pertubations of the hamiltonian at (i,j), and find the
    derivative dE/dH[i,j] to the first order by doing linear regression.
    """
    pertubations, square_errors = zip(
        *itertools.islice(
            generate_perturbed_square_errors(
                target_cov_matrix, hamiltonian, i, j, sigma
            ),
            n,
        )
    )

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        pertubations, square_errors
    )
    return slope


def stochastic_gradient(target_cov_matrix, hamiltonian, sigma, n):
    """
    find the stochastic derivative for each entry in the hamiltonian, and generate a gradient matrix.
    """
    gradient = np.zeros(hamiltonian.shape)
    for i, j in itertools.product(*map(range, hamiltonian.shape)):
        gradient[i, j] = stochastic_derivative(
            target_cov_matrix, hamiltonian, i, j, sigma, n
        )
    return gradient


def find_hamiltonian_for_cov_matrix(
    target_cov_matrix, initial_hamiltonian, sigma, n, eta
):
    """
    a generator performing gradient descent with learning rate eta, yielding the improved
    hamiltonian on each iteration.
    """
    hamiltonian = initial_hamiltonian.copy()
    while True:
        gradient = stochastic_gradient(target_cov_matrix, hamiltonian, sigma, n)
        hamiltonian = hamiltonian - eta * gradient
        yield hamiltonian


if __name__ == "__main__":
    iterations_done = 0
    cmatrix = np.loadtxt("path_to_cmatrix")
    diag = np.ones(cmatrix.shape[0] - 1)
    H_init = -np.diag(diag, +1) + np.diag(diag, -1)
    H_init[0, cmatrix.shape[0] - 1] = -1
    H_init[cmatrix.shape[0] - 1, 0] = 1

    for hamiltonian in find_hamiltonian_for_cov_matrix(
        cmatrix, H_init, sigma=1, n=100, eta=0.001
    ):
        iterations_done += 1
        sq_err = squared_error(cmatrix, hamiltonian)
        print(
            "The squared error for iteration {} is {}".format(iterations_done, sq_err)
        )

        if sq_err < 5:
            # we are there!
            print(hamiltonian)
            break

        if iterations_done > 100:
            break
