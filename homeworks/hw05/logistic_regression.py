#!/usr/bin/python

import numpy as np
import pytest

from homeworks.hw01.perceptron import (label_points, random_line,
        random_points)
from itertools import count

MONTE_CARLO_NUM_POINTS = 400
NUM_RUNS = 100
NUM_TRAINING_POINTS = 100
W_DIFF_STOP = 0.01
LEARNING_RATE = 0.01


def main():
    results = [run_experiment(NUM_TRAINING_POINTS) for _ in range(NUM_RUNS)]
    num_epochs, error = np.mean(results, axis=0)
    print('num_epochs={}, e_out={}'.format(num_epochs, error))


def run_experiment(N):
    """
    Runs an experiment with `N` random points.
    """
    f = random_line()
    sample_points = random_points(N)
    correct_labels = label_points(sample_points, f)
    g, num_epochs = log_reg_sgd_learning(sample_points, correct_labels)
    error = cross_entropy_e_out(g)
    return num_epochs, error


def log_reg_sgd_learning(points, correct_labels):
    w = np.zeros(3)
    for num_epochs in count(start=1, step=1):
        random_indexes = np.arange(points.shape[0])
        np.random.shuffle(random_indexes)
        w_before_epoch = w.copy()

        for i in random_indexes:
            y = correct_labels[i]
            x = np.hstack((1, points[i]))  # Add x0
            e_in = -y * x / (1 + np.exp(y * w.dot(x)))
            w -= LEARNING_RATE * e_in

        if np.linalg.norm(w - w_before_epoch) < 0.01:
            break

    return w, num_epochs


def cross_entropy_e_out(w):
    points = random_points(MONTE_CARLO_NUM_POINTS)
    num_points = points.shape[0]
    x = np.column_stack((np.ones(num_points), points))
    s = x.dot(w)
    y = np.exp(s) / (1 + np.exp(s))
    e_out = np.mean(np.log(1 + np.exp(-y * s)), axis=0)
    return e_out


if __name__ == '__main__':
    pytest.main(['-s', __file__])
    main()
