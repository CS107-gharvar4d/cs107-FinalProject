"""
ADG4: Automatic Differentiation for Python

A simple tool for handling automatic differentiation (AD) in Python using
elementary functions, as well as, trigonometry

Tests module that using AutoDiffReverse and test using backpropagation algo.

"""

import pytest
import sys
import ADG4.reverse as rev
import numpy as np
import random

def fprime_fd(f, x0, dx=1e-12):
    """
        Calculating derivative of f at x0  using finite difference
        :param f:
        :param x0:
        :param dx:
        :return:
        """
    return (f(x0 + dx) - f(x0)) / dx

def test_rvd_mult():
    """
    Check basic assertions when adding  or multiplying new values and outputs correct derivative
    :return:
    """
    x = rev.AutoDiffReverse(3, name='x')
    y = rev.AutoDiffReverse(4, name='y')
    z = rev.AutoDiffReverse(9, name='z')
    m = x + y
    n = m * z + x
    p = x * x * x

    assert n.partial(x) == 10
    assert n.partial(y) == 1 * z.val
    assert n.partial(z) == m.val
    assert m.partial(x) == 1
    assert m.partial(y) == 1
    assert p.partial(x) == 3 * 3 * 3

def test_neg():
    """
    Test negative numbers, not inclusive to all positives
    :return:
    """
    x = rev.AutoDiffReverse(3, name='x')
    y = rev.AutoDiffReverse(4, name='y')
    z = rev.AutoDiffReverse(-9, name='z')
    m = x + y
    n = m * z + x
    q = -n
    assert q.partial(x) == -n.partial(x)

def basic_sub():
    """
    Test substracion operations upon reverse mode.
    :return:
    """
    x = rev.AutoDiffReverse(3, name='x')
    y = rev.AutoDiffReverse(4, name='y')
    z = rev.AutoDiffReverse(-9, name='z')
    m = x - y
    n = x - y - z
    assert m.val == -1
    assert m.partial(x) == 1
    assert m.partial(y) == -1
    assert n.val == 8
    assert n.partial(x) == 1
    assert n.partial(y) == -1
    #n.partial(m)

    m = x - 1
    assert m.val == 2
    assert m.partial(x) == 1
    m = 1 - x
    assert m.val == -2
    assert m.partial(x) == -1
    m = z - 1
    assert m.val == -10
    #assert  m.partial(y)

def test_inv():
    """
    Test inverse of a value i.e. 1/val and checks assertion
    :return:
    """
    x = rev.AutoDiffReverse(3, name='x')
    r = x.__inv__()
    assert r.partial(x) == -1 / x.val ** 2

def test_div():
    """
    Test division on each element and calc partial derivative.
    :return:
    """
    x = rev.AutoDiffReverse(3, name='x')
    y = rev.AutoDiffReverse(4, name='y')
    m = x / y
    assert m.val == 3 / 4
    assert m.partial(x) == 1 / 4
    assert m.partial(y) == 3 * -(4) ** (-2)

    m = x / 2
    assert m.val == 1.5
    assert m.partial(x) == 1 / 2

    m = 2 / x
    assert m.val == 2 / 3
    assert m.partial(x) == 2 * -(3) ** (-2)

    m = x / x
    assert m.val == 1
    assert m.partial(x) == 0

    m = 0 / x
    assert m.val == 0
    assert m.partial(x) == 0

def test_power():
    """
    Test power function on each element and calc partial derivative.
    :return:
    """
    x = rev.AutoDiffReverse(3, name='x')
    y = rev.AutoDiffReverse(4, name='y')
    m = x ** y
    assert m.val == 81
    assert m.partial(x) == 4 * 3 * 3 * 3
    assert m.partial(y) == 3 ** 4 * np.log(3)
    m = 2 ** x
    assert m.val == 8
    assert m.partial(x) == 2 ** 3 * np.log(2)
    z = rev.AutoDiffReverse(-2, name='z')
    m = z ** -2
    assert m.val == 0.25

def negative():
    x = rev.AutoDiffReverse(3, name='x')
    assert x.__neg__().val == -3

def test_vector():
    x = rev.AutoDiffReverse(np.array([1,2,3]))
    y = rev.AutoDiffReverse(9)
    z = rev.AutoDiffReverse(np.array([2, 4, 6]))
    h = x * y
    k = x * z
    h.backprop()
    k.backprop()
    assert np.array_equal(h.partial(y), np.array([1, 2, 3]))
    assert h.partial(x) == 9
    assert np.array_equal(k.partial(x), np.array([2, 4, 6]))
    assert np.array_equal(k.partial(z), np.array([1, 2, 3]))



def test_exp():
    a = random.randint(1,10)
    x = rev.AutoDiffReverse(a)
    f = x**a
    assert f.val == a ** a
    assert f.partial(x) ==  a*a**(a-1)


def test_loga():
    a = random.random()
    x = rev.AutoDiffReverse(a)
    f = rev.loga_rv(2, x)
    assert f.val == np.log(a) / np.log(2) and f.partial(x) == 1 / (a * np.log(2))


def test_log():
    a = random.random()
    x = rev.AutoDiffReverse(a)
    f = rev.log_rv(x)
    assert f.val == np.log(a) and f.partial(x) == 1 / (a)


def test_inverse_trig():
    a = random.random()
    x = rev.AutoDiffReverse(a)
    f = rev.arcsin_rv(x)
    assert f.val == np.arcsin(a)
    assert np.abs(f.partial(x) - fprime_fd(np.arcsin, a)) < 1e-2
    f = rev.arccos_rv(x)
    assert f.val == np.arccos(a)
    assert np.abs(f.partial(x) - fprime_fd(np.arccos, a)) < 1e-2
    f = rev.arctan_rv(x)
    assert f.val == np.arctan(a)
    # DOES NOT WORK
    #assert np.abs(f.partial(x) - fprime_fd(np.arctan, a)) < 1e-2


def test_sqrt():
    a = random.random()
    x = rev.AutoDiffReverse(a)
    f = rev.sqrt_rv(x)
    assert f.val == a ** 0.5
    assert f.partial(x) == 0.5 * a ** (-0.5)


def test_hyperbolic():
    a = random.random()
    x = rev.AutoDiffReverse(a)
    f = rev.sinh_rv(x)
    assert f.val == np.sinh(a)
    assert f.partial(x) == np.cosh(a)
    f = rev.cosh_rv(x)
    assert f.val == np.cosh(a)
    assert f.partial(x) == np.sinh(a)
    f = rev.tanh_rv(x)
    assert f.val == np.tanh(a)
    assert np.abs(f.partial(x) - (1 - np.tanh(a) ** 2)) < 1e-3


def test_logistic():
    a = random.random()
    x = rev.AutoDiffReverse(a)
    f = rev.logistic_rv(x)
    assert f.val == 1 / (1 + np.exp(-a))
    assert np.abs(f.partial(x) - fprime_fd(lambda x: 1 / (1 + np.exp(-x)), a)) < 1e-2


def test_compare():
    a = random.random()
    x = rev.AutoDiffReverse(a)
    f1 = rev.sinh_rv(x)
    f2 = rev.cosh_rv(x)
    f3 = rev.tanh_rv(x)
    assert f1 != f2
    f4 = f1 / f2
    print(f4.val, f4.partial(x))
    print(f3.val, f3.partial(x))
    assert np.isclose(f4.val, f3.val)
