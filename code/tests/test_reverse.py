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
