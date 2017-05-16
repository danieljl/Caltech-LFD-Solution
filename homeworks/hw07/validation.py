#!/usr/bin/python

import numpy as np
import pytest


def main():
    training_set = np.loadtxt('in.dta')
    test_set = np.loadtxt('out.dta')

    k_choices = range(3, 8)
    errors = [run_experiment(training_set, test_set, k, swapped=False)
              for k in k_choices]
    E_vals, E_outs = zip(*errors)
    print('k={}, E_val={}'.format(k_choices[np.argmin(E_vals)], np.min(E_vals)))
    print('k={}, E_out={}'.format(k_choices[np.argmin(E_outs)], np.min(E_outs)))
    print('E_out for k={}: {}'.format(k_choices[np.argmin(E_vals)], E_outs[np.argmin(E_vals)]))

    # Training set and validatin set are swapped

    k_choices = range(3, 8)
    errors = [run_experiment(training_set, test_set, k, swapped=True)
              for k in k_choices]
    E_vals, E_outs = zip(*errors)
    print('k={}, E_val={}'.format(k_choices[np.argmin(E_vals)], np.min(E_vals)))
    print('k={}, E_out={}'.format(k_choices[np.argmin(E_outs)], np.min(E_outs)))


def run_experiment(training_set, test_set, k, swapped=False):
    X_training, y_training = training_set[:25, :2], training_set[:25, 2]
    X_validation, y_validation = training_set[25:, :2], training_set[25:, 2]
    X_test, y_test = test_set[:, :2], test_set[:, 2]

    if swapped:
        X_training, X_validation = X_validation, X_training
        y_training, y_validation = y_validation, y_training

    X_training = nonlinear_transformation(X_training, k)
    X_validation = nonlinear_transformation(X_validation, k)
    X_test = nonlinear_transformation(X_test, k)

    w = regression_learning(X_training, y_training)
    E_val = error(y_validation, classify(w, X_validation))
    E_out = error(y_test, classify(w, X_test))
    return E_val, E_out


def classify(w, points):
    num_points = points.shape[0]
    a = np.column_stack((np.ones(num_points), points))
    return np.sign(a.dot(w))


def regression_learning(points, correct_labels, lambda_=0.):
    num_points = points.shape[0]
    X = np.column_stack((np.ones(num_points), points))
    num_dims = X.shape[1]
    pseudo_inverse = np.linalg.inv(X.T.dot(X) + lambda_ * np.identity(num_dims)).dot(X.T)
    w = pseudo_inverse.dot(correct_labels)
    return w


def nonlinear_transformation(points, k):
    if k < 3 or k > 7:
        raise ValueError('k must be between 3 and 7, inclusive')

    points_T = points.T
    x1 = points_T[0]
    x2 = points_T[1]
    result = [points, x1 ** 2, x2 ** 2, x1 * x2, np.abs(x1 - x2),
              np.abs(x1 + x2)]
    return np.column_stack(result[:k - 1])


def error(y_actuals, y_preds):
    return np.mean(~np.isclose(y_actuals, y_preds))


if __name__ == '__main__':
    pytest.main(['-s', __file__])
    main()
