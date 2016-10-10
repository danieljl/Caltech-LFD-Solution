#!/usr/bin/python

import numpy as np
import pytest

NUM_RUNS = 100000
NUM_COINS = 1000
NUM_FLIPS = 10


def main():
    v = np.array([calculate_v(flip_coins(NUM_COINS, NUM_FLIPS))
                  for _ in range(NUM_RUNS)])
    v_mean = v.mean(axis=0)
    print('v_1={}, v_rand={}, v_min={}'.format(*v_mean))


def calculate_v(experiment):
    v_1 = np.mean(experiment[0])
    v_rand = np.mean(experiment[np.random.randint(experiment.shape[0])])
    v_min = np.min(np.mean(experiment, axis=1))
    return v_1, v_rand, v_min


def flip_coins(num_coins, num_flips):
    """
    Returns a matrix with the size of `num_coins` x `num_flips`.
    Elements in the matrix have 2 possible values: 1 (head) and 0 (tail).
    """
    return np.random.randint(2, size=(num_coins, num_flips))


# ========================================================================== #

def test_calculate_v():
    experiment = np.array([[0, 1], [1, 1], [0, 0]])
    v_1, v_rand, v_min = calculate_v(experiment)
    assert np.isclose(v_1, .5)
    assert .0 <= v_rand <= 1.
    assert np.isclose(v_min, .0)

# ========================================================================== #


if __name__ == '__main__':
    pytest.main(['-s', __file__])
    main()
