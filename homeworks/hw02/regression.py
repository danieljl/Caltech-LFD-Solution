import numpy as np
import pytest

from homeworks.hw01.perceptron import (calculate_p_f_neq_g,
        perceptron_learning, label_points, random_line, random_points)

NUM_RUNS = 1000
MONTE_CARLO_NUM_POINTS = 1000


def main():
    num_points = 100
    result = [run_experiment(num_points) for _ in range(NUM_RUNS)]
    E_in, E_out = np.mean(result, axis=0)
    print('E_in={}, E_out={}'.format(E_in, E_out))

    num_points = 10
    result = [run_experiment(num_points, run_pla=True)
              for _ in range(NUM_RUNS)]
    num_iterations = np.mean(result, axis=0)
    print('num_iterations={}'.format(num_iterations))


def run_experiment(N, run_pla=False):
    """
    Runs an experiment with `N` random points.
    """
    f = random_line()
    sample_points = random_points(N)
    correct_labels = label_points(sample_points, f)
    g = regression_learning(sample_points, correct_labels)

    if run_pla:
        g_perceptron, num_iterations = perceptron_learning(sample_points,
                correct_labels, w_init=g)
        return num_iterations
    else:
        E_in = calculate_p_f_neq_g(f, g, points=sample_points)
        E_out = calculate_p_f_neq_g(f, g, num_points=MONTE_CARLO_NUM_POINTS)
        return E_in, E_out


def regression_learning(points, correct_labels):
    num_points = points.shape[0]
    X = np.column_stack((np.ones(num_points), points))
    pseudo_inverse = np.linalg.inv(X.T.dot(X)).dot(X.T)
    w = pseudo_inverse.dot(correct_labels)
    return w


# ========================================================================== #

def test_regression_learning():
    f = random_line()
    sample_points = random_points(10)
    correct_labels = label_points(sample_points, f)
    g = regression_learning(sample_points, correct_labels)
    assert g.shape == (3,)

# ========================================================================== #


if __name__ == '__main__':
    pytest.main(['-s', __file__])
    main()
