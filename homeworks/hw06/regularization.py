#!/usr/bin/python

import numpy as np
import pytest


def main():
    training_set = np.loadtxt('in.dta')
    X_training = nonlinear_transformation(training_set[:, :2])
    y_training = training_set[:, 2]
    test_set = np.loadtxt('out.dta')
    X_test = nonlinear_transformation(test_set[:, :2])
    y_test = test_set[:, 2]

    print(run_experiment(X_training, y_training, X_test, y_test, lambda_=0.))
    print(run_experiment(X_training, y_training, X_test, y_test, lambda_=1e-3))
    print(run_experiment(X_training, y_training, X_test, y_test, lambda_=1e3))

    k_choices = range(-2, 3)
    E_outs = [run_experiment(X_training, y_training, X_test, y_test,
                             lambda_=10**k)[1]
              for k in k_choices]
    print(np.argmin(E_outs))

    k_choices = range(-10, 10)
    E_outs = [run_experiment(X_training, y_training, X_test, y_test,
                             lambda_=10**k)[1]
              for k in k_choices]
    print(np.min(E_outs))


def run_experiment(X_training, y_training, X_test, y_test, lambda_):
    w = regression_learning(X_training, y_training, lambda_=lambda_)
    E_in = error(y_training, classify(w, X_training))
    E_out = error(y_test, classify(w, X_test))
    return E_in, E_out


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


def nonlinear_transformation(points):
    points_T = points.T
    x1 = points_T[0]
    x2 = points_T[1]
    return np.column_stack((points, x1 ** 2, x2 ** 2, x1 * x2, np.abs(x1 - x2),
                           np.abs(x1 + x2)))


def error(y_actuals, y_preds):
    return np.mean(~np.isclose(y_actuals, y_preds))


if __name__ == '__main__':
    pytest.main(['-s', __file__])
    main()
