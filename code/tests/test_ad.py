import pytest
import sys
sys.path.append('../ADG4/')
import ad as ad 
import numpy as np
import random as random

def test_basic():
    #a = 3.0 # Value to evaluate at
    a = np.random.rand(1)
    #a.type()
    #print(a)
    #a=5.0
    x = ad.AutoDiffToy(a)
    
    alpha = 2.0
    beta = 3.0
    print(x.val,x.der)
    f =  alpha*x+beta
    print(x.val,x.der)
    print(f.val)
    print("??",a)
    
    assert f.val==alpha*a+beta and f.der==alpha
    print(f.val,f.der)
    print(x.val,x.der)
    f = x * alpha + beta
    print(f.val,f.der)
    assert f.val==alpha*a+beta and f.der==alpha
    #print(f.val, f.der)
    f = beta + alpha * x
    assert f.val==alpha*a+beta and f.der==alpha
    #print(f.val, f.der) 
    f = beta + x * alpha
    assert f.val==alpha*a+beta and f.der==alpha
    #print(f.val, f.der) 
    f = beta - x * alpha
    #print(f.val, f.der)
    assert f.val==beta-a*alpha and f.der==-alpha
    
##########
###Boer Nov17
def test_trig():
    a = random.random()
    print(a)
    x=ad.AutoDiffToy(a)
    f = ad.sin_ad(x)
    #print(f.val, f.der)
    assert f.val==np.sin(a) and f.der==np.cos(a)
    f1= ad.cos_ad(x)
    assert f1.val==np.cos(a) and f1.der==-np.sin(a)
    f1= ad.tan_ad(x)
    assert f1.val==np.tan(a) and np.abs(f1.der-1/(np.cos(a)**2))<1e-5

###Boer Nov17
def test_exp():
    a = random.random()
    print(a)
    x=ad.AutoDiffToy(a)
    f=ad.exp_ad(x)
    assert f.val==np.exp(a) and f.der==np.exp(a)
###########
###Boer Nov17
def test_pow():
    #a = 2.0 # Value to evaluate at    
    a = np.random.rand(1)
    x = ad.AutoDiffToy(a)
    f1= 2**x
    print(f1.val,f1.der)
    assert f1.val==2**a and np.abs(f1.der-2**a*np.log(2))<1e-5    
    f2= x**2
    assert f2.val==a**2 and f2.der==2*a
    f3= x**x
    assert f3.val==a**a and np.abs(f3.der-(np.log(a)+1)*a**a)<1e-5
    f4= x.__rpow__(x)
    assert f4.val==a**a and np.abs(f4.der-(np.log(a)+1)*a**a)<1e-5
###########
##Boer Nov17
def test_twoAD():
    a = 2.0 # Value to evaluate at
    x = ad.AutoDiffToy(a)
    f1= 2*x
    f2=f1+x
    #print(f2.val,f2.der)
    f3=f1-x
    f4=f1*x
    print(f3.val,f3.der)
    print(f4.val,f4.der)
    assert f1.val==4.0 and f1.der==2.0
    assert f2.val==6.0 and f2.der==3
    assert f3.val==2 and f3.der==1
    assert f4.val==8 and f4.der==8
 
