#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import scipy
from gradient_descent import *
from cov_matrix import *
from matplotlib import pyplot as plt
from matplotlib import rc


def perturb(Hamiltonian, x, i):
    H = Hamiltonian.copy()
    j = (i + 1) % true_cmatrix.shape[0]
    H[i, j] = H[i, j] + x
    H[j, i] = H[j, i] - x
    return H


def show_pertubation(x, i):
    H = perturb(H_init, x, i)
    print("Hamiltonian:")
    print(H)
    cmatrix = groundstate_correlation_matrix(H)
    plt.title("Perturbed cmatrix - original cmatrix")
    r = plt.imshow(cmatrix - true_cmatrix, cmap="seismic")
    plt.colorbar()
    return r


def approximate_parent_hamiltonian(true_cmatrix):
    diag = np.random.rand(true_cmatrix.shape[0] - 1)
    H_candidate = np.diag(diag, +1) - np.diag(diag, -1)
    H_candidate[0, true_cmatrix.shape[0] - 1] = -1
    H_candidate[true_cmatrix.shape[0] - 1, 0] = 1

    basis = np.array(
        [
            (
                groundstate_correlation_matrix(perturb(H_candidate, 1, i))
                - groundstate_correlation_matrix(H_candidate)
            ).flatten()
            for i in range(H_candidate.shape[0])
        ]
    )

    target_diff = (groundstate_correlation_matrix(H_candidate) - true_cmatrix).flatten()
    coeff, _, _, _ = scipy.linalg.lstsq(basis.T, target_diff)

    H_candidate_diff = np.diag(coeff[:-1], +1) - np.diag(coeff[:-1], -1)
    H_candidate_diff[0, len(coeff) - 1] = +coeff[-1]
    H_candidate_diff[len(coeff) - 1, 0] = -coeff[-1]

    return H_candidate - H_candidate_diff


if __name__ == "__main__":

    rc("text", usetex=True)
    rc("font", family="serif")

    np.set_printoptions(precision=2)

    H_init = fibonacci_hamiltonian(0.2, 1.2, 7)
    true_cmatrix = groundstate_correlation_matrix(H_init)

    H_parent = approximate_parent_hamiltonian(true_cmatrix)

    fig = plt.figure(figsize=(2.9, 2.9))
    fig = plt.figure(figsize=(2.9, 2.9))
    ax = fig.gca()
    ax.matshow(H_init, cmap="seismic")
    ax.set_xlabel("$i$")
    ax.set_ylabel("$j$")
    plt.tight_layout()
    fig.savefig("output/test_target.pdf")

    fig = plt.figure(figsize=(2.9, 2.9))
    plt.matshow(H_parent, cmap="seismic")
    plt.xlabel("$i$")
    plt.ylabel("$j$")
    plt.tight_layout()
    plt.savefig("output/test_approx.pdf")
