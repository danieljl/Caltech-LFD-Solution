#!/usr/bin/python

import numpy as np
import pytest

from cvxopt import matrix, solvers
from homeworks.hw01.perceptron import (calculate_p_f_neq_g,
        perceptron_learning, label_points, random_line, random_points)

NUM_RUNS = 1000
MONTE_CARLO_NUM_POINTS = 1000

solvers.options['show_progress'] = False  # Disable cvxopt's logging


def main():
    for N in (10, 100):
        results = [run_experiment(N) for _ in range(NUM_RUNS)]
        svm_better, num_supports = np.mean(results, axis=0)
        print('N={}: percentage={}, supports={}'.format(N, svm_better,
                                                        num_supports))


def run_experiment(N):
    f = random_line()
    sample_points = random_points(N)
    correct_labels = label_points(sample_points, f)
    g_pla, num_iterations = perceptron_learning(sample_points, correct_labels)
    g_svm, num_supports = svm_learning(sample_points, correct_labels)

    montel_carlo_points = random_points(MONTE_CARLO_NUM_POINTS)
    E_out_pla = calculate_p_f_neq_g(f, g_pla, points=montel_carlo_points)
    E_out_svm = calculate_p_f_neq_g(f, g_svm, points=montel_carlo_points)

    is_svm_better = 1. if E_out_svm < E_out_pla else 0.
    # pytest.set_trace()
    return is_svm_better, num_supports


def svm_learning(sample_points, correct_labels):
    correct_labels = correct_labels.astype(float)
    num_points = sample_points.shape[0]

    xx = sample_points.dot(sample_points.T)
    y = correct_labels.reshape((num_points, 1))
    yy = y.dot(y.T)
    P = xx * yy

    q = np.full((num_points,), -1.)
    G = np.diag(np.full((num_points,), -1.))
    h = np.zeros((num_points,))
    A = correct_labels.reshape((1, num_points)).copy()
    b = np.zeros((1,))
    solution = solvers.qp(*map(matrix, [P, q, G, h, A, b]))
    alpha = np.array(solution['x']).flatten()

    alpha_y = (alpha * correct_labels).reshape(num_points, 1)
    w = np.sum(alpha_y * sample_points, axis=0)
    is_support = ~np.isclose(alpha, np.zeros((num_points,)))
    num_supports = np.sum(is_support)
    idx_support = np.argmax(is_support)
    y_support = correct_labels[idx_support]
    x_support = sample_points[idx_support]
    b_support = 1. / y_support - w.dot(x_support)
    w = np.hstack((b_support, w))

    # TODO Check if w seperates sample_points perfectly

    # TODO Check constraints
    if not np.alltrue(G.dot(alpha) <= h):
        print(G.dot(alpha))
        print(h)
        pytest.set_trace()
    if not np.allclose(A.dot(alpha), b):
        print('wowwwwwwwwwwww')
        print(A.dot(alpha))
        print(b)
        pytest.set_trace()

    return w, num_supports


# ========================================================================== #

def test_svm_training():
    points = np.array([[0., 4.1], [1., 1.9], [2., .1]])
    labels = np.array([-1, 1, -1])
    w, num_supports = svm_learning(points, labels)
    w /= w[0]  # Normalize w so that w0 = 1
    expected = np.array([1., -.5, -.25])
    assert np.allclose(w, expected, atol=.01)

# ========================================================================== #


if __name__ == '__main__':
    pytest.main(['-s', __file__])
    main()
