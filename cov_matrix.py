#!/usr/bin/env python3
from colorama import Fore, Back, Style
import itertools
import numpy as np
import operator
import scipy
import unittest


def fibonacci_sequence(cutoff, a=0, b=1):
    """
    Create a fibonacci sequence, using a as a seed.

    Parameters
    ==========

    cuttoff: number of iterations
    a: first parameter
    b: second parameter
    """
    sequence = [a]

    for i in range(0, cutoff):
        duplicate = []
        for n in range(0, len(sequence)):
            if sequence[n] == a:
                duplicate.extend([a, b])
            elif sequence[n] == b:
                duplicate.append(a)
            else:
                print("oops something went wrong, is not a or b")
                break
        sequence = duplicate
    return sequence


def alternating(a, b):
    """
    A generator for alternating sequences (a, b, a, b, a, b, a, ...)
    """
    while True:
        yield a
        yield b


def kitaev_hamiltonian(size, j1, j2):
    """
    Constructs the hamiltonian matrix for a kitaev chain.
    """
    # size - 1 so that end == start
    offdiagonal = list(itertools.islice(alternating(j1, j2), size - 1))
    return np.diag(offdiagonal, 1) - np.diag(offdiagonal, -1)


def fibonacci_hamiltonian(a, b, length_sequence):
    """
    Constructs the hamiltonian matrix for a fibonacci sequence.
    """
    sequence = fibonacci_sequence(length_sequence, a, b)
    last_el = sequence.pop()
    H = np.diag(sequence, 1) - np.diag(sequence, -1)
    H[0, H.shape[1] - 1] = last_el
    H[H.shape[0] - 1, 0] = -last_el
    return H


def alternating_fibonacci_hamiltonian(a, b, length_sequence):
    """
    Constructs the hamiltonian matrix for an alternating fibonacci sequence.
    """
    sequence = fibonacci_sequence(length_sequence, a, b)
    last_el = sequence.pop()
    off_diagonal = [1]

    for i in range(0, len(sequence)):
        off_diagonal.append(sequence[i])
        off_diagonal.append(1)

    H = np.diag(off_diagonal, 1) - np.diag(off_diagonal, -1)
    H[0, H.shape[1] - 1] = last_el
    H[H.shape[0] - 1, 0] = -last_el
    return H


def random_sequence_hamiltonian(sequence):
    """
    Constructs the hamiltonian matrix for a random sequence.
    """
    last_el = sequence.pop()
    H = np.diag(sequence, 1) - np.diag(sequence, -1)
    H[0, H.shape[1] - 1] = last_el
    H[H.shape[0] - 1, 0] = -last_el
    return H


def vacuum_correlation_matrix(size):
    return kitaev_hamiltonian(size, -1, 0)


def ordered_eigenspectrum(M):
    """
    Find the eigenspectrum of the matrix M, and return the eigenvectors and
    eigenvalues sorted by the eigenvalues.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues, eigenvectors = zip(
        *sorted(zip(eigenvalues, eigenvectors.T), key=lambda pair: pair[0])
    )
    return eigenvalues, eigenvectors


def normalized(vector):
    """
    Return an L2-Normalized vector.
    """
    return (1.0 / np.linalg.norm(vector)) * np.array(vector)


def transformation_matrix(hamiltonian):
    """
    Build the transformation matrix Q to go from the vacuum correlation matrix
    to the ground correlation matrix
    """
    eigenvectors = (
        eigenvector
        for (eigenvalue, eigenvector) in zip(*ordered_eigenspectrum(1j * hamiltonian))
        if eigenvalue > 0
    )

    # The following generator yields the real part and then the imaginary part
    # of each eigenvector.
    def real_imag():
        for eigenvector in eigenvectors:
            yield normalized(eigenvector.real)
            yield normalized(eigenvector.imag)

    return np.array(list(real_imag()))


def groundstate_correlation_matrix(hamiltonian):
    """
    Find the groundstate correlation matrix from the hamiltonian and the transformation matrix Q.
    """

    Q = transformation_matrix(hamiltonian)

    # hamiltonian is a square matrix, get only one side.
    size, _ = hamiltonian.shape

    # Using Q.T = Q^(-1), diagonalize w.r.t. Q
    return -Q.T @ vacuum_correlation_matrix(size) @ Q


class TestStuff(unittest.TestCase):
    def test_normalized(self):
        # use allclose, since 0.6 isn't exactly representable in a float64.
        self.assertTrue(np.allclose(normalized([0, 3, 4]), [0.0, 0.6, 0.8]))

    def test_alternating(self):
        sequence = list(itertools.islice(alternating(42, 23), 6))
        should_be = [42, 23, 42, 23, 42, 23]
        self.assertEqual(sequence, should_be)

    def test_kitaev_hamiltonian(self):
        h = [
            [0, 1, 0, 0, 0, 0],
            [-1, 0, 2, 0, 0, 0],
            [0, -2, 0, 1, 0, 0],
            [0, 0, -1, 0, 2, 0],
            [0, 0, 0, -2, 0, 1],
            [0, 0, 0, 0, -1, 0],
        ]
        self.assertTrue(np.array_equal(h, kitaev_hamiltonian(6, 1, 2)))

    def test_vacuum_correlation_matrix(self):
        f = [
            [0, -1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 1, 0],
        ]
        self.assertTrue(np.array_equal(f, vacuum_correlation_matrix(6)))

    def test_ordered_eigenspectrum(self):
        M = np.array([[3, 1, 1], [0, 2, 1], [0, 0, 1]])
        eigenvalues, eigenvectors = ordered_eigenspectrum(M)

        # eigenvalues are sorted.
        self.assertTrue(np.array_equal(eigenvalues, [1, 2, 3]))

        # eigenvalues and eigenvectors match.
        for eigenvector, eigenvalue in zip(eigenvectors, eigenvalues):
            self.assertTrue(
                np.array_equal(M.dot(eigenvector), eigenvalue * eigenvector)
            )

    def test_correlation_matrix(self):
        Gamma = [
            [0, 0.600267, 0, 0.413519, 0, 0.684603],
            [-0.600267, 0, 0.787015, 0, 0.142435, 0],
            [0, -0.787015, 0, 0.457832, 0, 0.413519],
            [-0.413519, 0, -0.457832, 0, 0.787015, 0],
            [0, -0.142435, 0, -0.787015, 0, 0.600267],
            [-0.684603, 0, -0.413519, 0, -0.600267, 0],
        ]
        h = kitaev_hamiltonian(6, 1, 2)
        self.assertTrue(np.allclose(groundstate_correlation_matrix(h), Gamma))


def check_singular(h):
    """
    Check whether a hamiltonian h is singular or not.
    """

    size = h.shape[0]
    rank = np.linalg.matrix_rank(h)
    if h.shape == (rank, rank):
        print(f"{Fore.GREEN}non-singular{Style.RESET_ALL}: size={size} rank={rank}")
        return False
    else:
        print(
            f"{Fore.RED}singular{Style.RESET_ALL}: size={size} rank={Fore.YELLOW}{rank}{Style.RESET_ALL} h.shape={h.shape}"
        )
        return True


def find_singular_hamiltonians(
    hamiltonian=fibonacci_hamiltonian,
    length_parameter="length_sequence",
    limit=17,
    *args,
    **kwargs,
):
    assert limit <= 17
    for s in range(0, limit):
        kwargs[length_parameter] = s
        h = hamiltonian(*args, **kwargs)
        check_singular(h)
