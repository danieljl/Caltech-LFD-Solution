#!/usr/bin/python

import numpy as np
import pytest


def main():
    X_train_ori, y_train_ori = read_dataset('train')
    X_test_ori, y_test_ori = read_dataset('test')

    E_ins = []
    digits = np.arange(5, 10, dtype=float)
    for digit in digits:
        X_train, y_train = one_vs_all(X_train_ori, y_train_ori, digit)
        X_test, y_test = one_vs_all(X_test_ori, y_test_ori, digit)
        E_in, _ = run_experiment(X_train, y_train, X_test, y_test, lambda_=1.)
        E_ins.append(E_in)
    digit_lowest = digits[np.argmin(E_ins)]
    print('The lowest E_in achieves when digit={}'.format(digit_lowest))

    E_outs = []
    digits = np.arange(0, 5, dtype=float)
    for digit in digits:
        X_train, y_train = one_vs_all(X_train_ori, y_train_ori, digit)
        X_test, y_test = one_vs_all(X_test_ori, y_test_ori, digit)
        X_train = nonlinear_transform(X_train)
        X_test = nonlinear_transform(X_test)
        _, E_out = run_experiment(X_train, y_train, X_test, y_test, lambda_=1.)
        E_outs.append(E_out)
    digit_lowest = digits[np.argmin(E_outs)]
    print('The lowest E_out achieves when digit={}'.format(digit_lowest))

    print('\n--------------------\n')

    E_ins, E_outs = [], []
    digits = np.array([0., 9., 5.])
    for digit in digits:
        for is_transformed in [False, True]:
            X_train, y_train = one_vs_all(X_train_ori, y_train_ori, digit)
            X_test, y_test = one_vs_all(X_test_ori, y_test_ori, digit)
            if is_transformed:
                X_train = nonlinear_transform(X_train)
                X_test = nonlinear_transform(X_test)
            E_in, E_out = run_experiment(X_train, y_train, X_test, y_test,
                                         lambda_=1.)
            print('digit={}: E_in={}, E_out={}, is_transformed={}'.format(
                  digit, E_in, E_out, is_transformed))

    print('\n--------------------\n')

    E_ins, E_outs = [], []
    for lambda_ in [1., .01]:
        X_train, y_train = one_vs_one(X_train_ori, y_train_ori, 1., 5.)
        X_test, y_test = one_vs_one(X_test_ori, y_test_ori, 1., 5.)
        X_train = nonlinear_transform(X_train)
        X_test = nonlinear_transform(X_test)
        E_in, E_out = run_experiment(X_train, y_train, X_test, y_test,
                                     lambda_=lambda_)
        print('lambda={}: E_in={}, E_out={}'.format(
              lambda_, E_in, E_out))


def read_dataset(dataset_type):
    dataset = np.loadtxt('features.' + dataset_type)
    return dataset[:, 1:], dataset[:, 0]  # X, y


def one_vs_all(X, y, one):
    X, y = X.copy(), y.copy()
    one_idxs = np.isclose(y, one)
    y[one_idxs] = 1.
    y[~one_idxs] = -1.
    return X, y


def one_vs_one(X, y, one_1, one_2):
    X, y = X.copy(), y.copy()
    is_one_1 = np.isclose(y, one_1)
    is_one_2 = np.isclose(y, one_2)
    y[is_one_1] = 1.
    y[is_one_2] = -1.
    X = X[is_one_1 | is_one_2]
    y = y[is_one_1 | is_one_2]

    return X, y


def run_experiment(X_train, y_train, X_test, y_test, lambda_):
    w = regression_learning(X_train, y_train, lambda_=lambda_)
    E_in = error(y_train, classify(w, X_train))
    E_out = error(y_test, classify(w, X_test))
    return E_in, E_out


def regression_learning(points, correct_labels, lambda_=0.):
    X = add_w0(points)
    num_dims = X.shape[1]
    pseudo_inverse = np.linalg.inv(X.T.dot(X) +
                                   lambda_ * np.identity(num_dims)).dot(X.T)
    w = pseudo_inverse.dot(correct_labels)
    return w


def classify(w, points):
    a = add_w0(points)
    return 1. * np.sign(a.dot(w))


def add_w0(points):
    num_points = points.shape[0]
    return np.column_stack((np.ones(num_points), points))


def nonlinear_transform(points):
    points_T = points.T
    x1 = points_T[0]
    x2 = points_T[1]
    return np.column_stack((points, x1 * x2, x1 ** 2, x2 ** 2))


def error(y_actuals, y_preds):
    return np.mean(~np.isclose(y_actuals, y_preds))


if __name__ == '__main__':
    main()
