#!/usr/bin/python

import itertools
import numpy as np
import pytest

from scipy import stats
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC

GAMMA = 1.
COEF0 = 1.


def main():
    # polynomial()
    cross_validation()
    # rbf()


def read_dataset(dataset_type):
    dataset = np.loadtxt('features.' + dataset_type)
    return dataset[:, 1:], dataset[:, 0]  # X, y


def polynomial():
    clf = SVC(C=.01, kernel='poly', degree=2, gamma=GAMMA, coef0=COEF0)
    for offset in [0, 1]:
        num_supports, E_ins = [], []
        digits = np.array([0, 2, 4, 6, 8], dtype=float) + offset
        for digit in digits:
            X_training, y_training = read_dataset('train')
            y_training[~np.isclose(y_training, digit)] = -1.
            clf.fit(X_training, y_training)
            E_ins.append(1 - clf.score(X_training, y_training))
            num_supports.append(clf.n_support_.sum())

        chosen_idx = np.argmax(E_ins) if offset == 0 else np.argmin(E_ins)
        print('digit={}: E_in={}, num_supports={}'.format(digits[chosen_idx],
                E_ins[chosen_idx], num_supports[chosen_idx]))

    print('\n--------------------\n')

    X_training, y_training = read_dataset('train')
    one_or_five = np.isclose(y_training, 1.) | np.isclose(y_training, 5.)
    X_training, y_training = X_training[one_or_five], y_training[one_or_five]
    X_test, y_test = read_dataset('test')
    one_or_five = np.isclose(y_test, 1.) | np.isclose(y_test, 5.)
    X_test, y_test = X_test[one_or_five], y_test[one_or_five]

    Cs = [.001, .01, .1, 1.]
    clfs = [SVC(C=C, kernel='poly', degree=2, gamma=GAMMA, coef0=COEF0)
            for C in Cs]
    [clf.fit(X_training, y_training) for clf in clfs]
    num_supports = [clf.n_support_.sum() for clf in clfs]
    E_ins = [1 - clf.score(X_training, y_training) for clf in clfs]
    E_outs = [1 - clf.score(X_test, y_test) for clf in clfs]
    print('num_supports={}'.format(num_supports))
    print('E_ins={}'.format(E_ins))
    print('diff E_ins={}'.format(np.diff(E_ins, 1)))
    print('E_outs={}'.format(E_outs))
    print('diff E_outs={}'.format(np.diff(E_outs, 1)))

    print('\n--------------------\n')

    Cs = [.0001, .001, .01, 1]
    degrees = [2, 5]
    clfs = {C: {degree: SVC(C=C, kernel='poly', degree=degree, gamma=GAMMA,
                            coef0=COEF0).fit(X_training, y_training)
                for degree in degrees}
            for C in Cs}

    E_ins = [1 - clf.score(X_training, y_training)
             for clf in clfs[.0001].values()]
    print('C=0.0001: E_ins={}'.format(E_ins))

    num_supports = [clf.n_support_.sum() for clf in clfs[.001].values()]
    print('C=0.001: num_supports={}'.format(num_supports))

    E_ins = [1 - clf.score(X_training, y_training)
             for clf in clfs[.01].values()]
    print('C=0.01: E_ins={}'.format(E_ins))

    E_outs = [1 - clf.score(X_test, y_test)
              for clf in clfs[1].values()]
    print('C=1: E_outs={}'.format(E_outs))


def cross_validation():
    X_training, y_training = read_dataset('train')
    one_or_five = np.isclose(y_training, 1.) | np.isclose(y_training, 5.)
    X_training, y_training = X_training[one_or_five], y_training[one_or_five]
    Cs = [.0001, .001, .01, .1, 1.]
    clfs = [GridSearchCV(SVC(kernel='poly', degree=2, gamma=GAMMA, coef0=COEF0),
                         param_grid=dict(C=Cs),
                         cv=KFold(n_splits=10, shuffle=True),
                         n_jobs=8).fit(X_training, y_training)
            for _ in range(100)]
    chosen_Cs = [clf.best_params_['C'] for clf in clfs]
    E_cvs = [1 - clf.best_score_ for clf in clfs]
    print(stats.mode(chosen_Cs))
    print(np.mean(E_cvs))


def rbf():
    X_training, y_training = read_dataset('train')
    one_or_five = np.isclose(y_training, 1.) | np.isclose(y_training, 5.)
    X_training, y_training = X_training[one_or_five], y_training[one_or_five]
    X_test, y_test = read_dataset('test')
    one_or_five = np.isclose(y_test, 1.) | np.isclose(y_test, 5.)
    X_test, y_test = X_test[one_or_five], y_test[one_or_five]

    Cs = [.01, 1, 100, 1e4, 1e6]
    clfs = [SVC(C=C, kernel='rbf', gamma=GAMMA).fit(X_training, y_training)
            for C in Cs]
    E_ins = [1 - clf.score(X_training, y_training) for clf in clfs]
    print('E_ins={}'.format(E_ins))
    print('argmin E_ins={}'.format(np.argmin(E_ins)))
    E_outs = [1 - clf.score(X_test, y_test) for clf in clfs]
    print('E_outs={}'.format(E_outs))
    print('argmin E_outs={}'.format(np.argmin(E_outs)))


if __name__ == '__main__':
    main()
