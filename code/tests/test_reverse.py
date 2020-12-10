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

def test_add_mult():
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

