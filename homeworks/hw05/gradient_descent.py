#!/usr/bin/python

from itertools import count
from math import exp

LEARNING_RATE = 0.1
U_INIT, V_INIT = 1., 1.
ERROR_MIN = 1e-14


def main():
    coordinate_descent(max_steps=30)


def gradient_descent():
    u, v = U_INIT, V_INIT
    for iter_num in count(start=0):
        error = E(u, v)
        if error < ERROR_MIN:
            break

        du, dv = dE_du(u, v), dE_dv(u, v)
        u, v = u - LEARNING_RATE * du, v - LEARNING_RATE * dv

    print('iter_num={}, error={}, u={}, v={}'.format(iter_num, error, u, v))


def coordinate_descent(max_steps):
    u, v = U_INIT, V_INIT
    for iter_num in range(max_steps):
        error = E(u, v)
        if error < ERROR_MIN:
            break

        if iter_num % 2 == 0:
            du = dE_du(u, v)
            u -= LEARNING_RATE * du
        else:
            dv = dE_dv(u, v)
            v -= LEARNING_RATE * dv

    print('iter_num={}, error={}, u={}, v={}'.format(iter_num, error, u, v))


def E(u, v):
    return (u * exp(v) - 2. * v * exp(-u)) ** 2.


def dE_du(u, v):
    return 2. * (exp(v) + 2. * v * exp(-u)) * (u * exp(v) - 2. * v * exp(-u))


def dE_dv(u, v):
    return 2. * (u * exp(v) - 2. * exp(-u)) * (u * exp(v) - 2. * v * exp(-u))


if __name__ == '__main__':
    main()
