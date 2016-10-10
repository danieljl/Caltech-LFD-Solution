import numpy as np
import pytest

from homeworks.hw01 import perceptron
from homeworks.hw01.perceptron import label_points, random_points
from homeworks.hw02.regression import regression_learning

NUM_RUNS = 1000
MONTE_CARLO_NUM_POINTS = 1000
NUM_POINTS = 1000
FRACTION_NOISE = .1


def main():
    result = [run_experiment(NUM_POINTS) for _ in range(NUM_RUNS)]
    g, E_in, E_out = np.mean(result, axis=0)
    print('E_in={}'.format(E_in))

    result = [run_experiment(NUM_POINTS, nonlinear_transform=True)
              for _ in range(NUM_RUNS)]
    g, E_in, E_out = np.mean(result, axis=0)
    print('g={}, E_out={}'.format(g, E_out))

    g_choices = np.array([[-1., -.05, .08, .13, 1.5, 1.5],
                          [-1., -.05, .08, .13, 1.5, 15.],
                          [-1., -.05, .08, .13, 15., 1.5],
                          [-1., -1.5, .08, .13, .05, .05],
                          [-1., -.05, .08, 1.5, .15, .15]])
    diff_min, g_chosen, g_idx_chosen = None, None, None
    monte_carlo_points = random_points(MONTE_CARLO_NUM_POINTS)
    monte_carlo_points = transform_nonlinearly(monte_carlo_points)
    for idx, g_current in enumerate(g_choices):
        diff = perceptron.calculate_p_f_neq_g(g, g_current,
                                              points=monte_carlo_points)
        diff_min = diff if diff_min is None else diff_min
        g_chosen = g_current if g_chosen is None else g_chosen
        g_idx_chosen = idx if g_idx_chosen is None else g_idx_chosen
        if diff < diff_min:
            diff_min, g_chosen, g_idx_chosen = diff, g_current, idx
    print('diff_min={}, g_chosen={}, g_idx_chosen={}'.format(diff_min,
            g_chosen, g_idx_chosen))


def run_experiment(N, nonlinear_transform=False):
    """
    Runs an experiment with `N` random points.
    """
    sample_points = random_points(N)
    correct_labels = label_points_with_f(sample_points)

    if nonlinear_transform:
        sample_points_transformed = transform_nonlinearly(sample_points)
    else:
        sample_points_transformed = sample_points

    g = regression_learning(sample_points_transformed, correct_labels)
    E_in = calculate_p_f_neq_g(g, nonlinear_transform, points=sample_points)
    E_out = calculate_p_f_neq_g(g, nonlinear_transform,
                                num_points=MONTE_CARLO_NUM_POINTS)
    return g, E_in, E_out


def calculate_p_f_neq_g(g, nonlinear_transform, points=None, num_points=None):
    if points is None and num_points is None:
        raise ValueError('points and num_points cannot be both None')
    if points is None:
        points = random_points(num_points)
    if nonlinear_transform:
        points_transformed = transform_nonlinearly(points)
    else:
        points_transformed = points
    label_f = label_points_with_f(points)
    label_g = label_points(points_transformed, g)
    return np.mean(~np.isclose(label_f, label_g))


def label_points_with_f(points):
    x = points.T
    correct_labels = np.sign(x[0] ** 2 + x[1] ** 2 - 0.6)
    return make_noise(correct_labels, fraction=FRACTION_NOISE)


def make_noise(correct_labels, fraction):
    num_labels = correct_labels.shape[0]
    num_noise = int(round(fraction * num_labels))
    index = np.arange(num_labels)
    noise_index = np.random.choice(index, num_noise, replace=False)
    noisy_labels = correct_labels.copy()
    noisy_labels[noise_index] *= -1
    return noisy_labels


def transform_nonlinearly(points):
    points_T = points.T
    x1x2 = points_T[0] * points_T[1]
    x1x1 = points_T[0] ** 2
    x2x2 = points_T[1] ** 2
    points_transformed = np.column_stack((points, x1x2, x1x1, x2x2))
    return points_transformed


# ========================================================================== #

def test_make_noise():
    noise_fraction = .2
    correct_labels = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    noisy_labels = make_noise(correct_labels, fraction=noise_fraction)
    assert np.isclose(np.mean(correct_labels != noisy_labels),
                      noise_fraction)

# ========================================================================== #


if __name__ == '__main__':
    pytest.main(['-s', __file__])
    main()
