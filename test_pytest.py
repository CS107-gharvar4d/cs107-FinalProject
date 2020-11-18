import pytest
import sys
sys.path.append('./code')
import ad as ad 
from functools import reduce
import numpy as np


def test_xsquare():
    a = 2.0 # Value to evaluate at
    x = ad.AutoDiffToy(a)

    alpha = 2.0
    beta = 3.0

    f = alpha * x + beta
    
    #print(f.val, f.der)
    assert f.val==7.0 and f.derivs[x]==2.0
    f = x * alpha + beta
    assert f.val==7.0 and f.derivs[x]==2.0
    #print(f.val, f.der)
    f = beta + alpha * x
    assert f.val==7.0 and f.derivs[x]==2.0
    #print(f.val, f.der) 
    f = beta + x * alpha
    assert f.val==7.0 and f.derivs[x]==2.0
    #print(f.val, f.der) 
    f = beta - x * alpha
    #print(f.val, f.der)
    assert f.val==-1.0 and f.derivs[x]==-2.0
    f = ad.sin_ad(x)
    #print(f.val, f.der)
    assert f.val==0.9092974268256817 and f.derivs[x]==-0.4161468365471424 

def test_two_vars():
    a = 2.0 # Value to evaluate at
    x = ad.AutoDiffToy(a)
    b = 3.0
    y = ad.AutoDiffToy(b)

    f = x*y

    assert f.val == 6
    assert f.derivs[y] == 2
    assert f.derivs[x] == 3

def test_chain_vars():
    x = ad.AutoDiffToy(10)
    y = ad.AutoDiffToy(3)
    z =  ad.AutoDiffToy(5)

    f = 3*x + 2*x + y**3
    g = f - z

    assert g.val == 72
    assert g.derivs[x] == 5 # aka 3+2
    assert g.derivs[y] == 3 * 3 ** 2


@pytest.mark.skip(reason="Useful for benchmarking large scale functions, not needed every time")
def test_many_vars():
    ad_vars = [ad.AutoDiffToy(i) for i in range(10000)]

    f = reduce(lambda x, y: x*y, ad_vars)

    print(f.val)

def test_vector_input():
    x = ad.AutoDiffToy(np.array([1,2,3,5]))

    f = x * 3

    print(f.val)
    assert f.derivs[x] == 3

def test_vector_input():
    x = ad.AutoDiffToy(np.array([1,2,3,5]))
    y = ad.AutoDiffToy(np.array([1,1,1,1]))

    f = x * y

    print(f.val)
    assert all([a==b for a in f.derivs[x] for b in y.val])



