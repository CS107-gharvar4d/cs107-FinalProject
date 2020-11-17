import pytest
import sys
sys.path.append('./code')
import ad as ad 

def test_xsquare():
    a = 2.0 # Value to evaluate at
    x = ad.AutoDiffToy(a)

    alpha = 2.0
    beta = 3.0

    f = alpha * x + beta
    
    #print(f.val, f.der)
    assert f.val==7.0 and f.der==2.0
    f = x * alpha + beta
    assert f.val==7.0 and f.der==2.0
    #print(f.val, f.der)
    f = beta + alpha * x
    assert f.val==7.0 and f.der==2.0
    #print(f.val, f.der) 
    f = beta + x * alpha
    assert f.val==7.0 and f.der==2.0
    #print(f.val, f.der) 
    f = beta - x * alpha
    #print(f.val, f.der)
    assert f.val==-1.0 and f.der==-2.0
    f = ad.sin_ad(x)
    #print(f.val, f.der)
    assert f.val==0.9092974268256817 and f.der==-0.4161468365471424 

