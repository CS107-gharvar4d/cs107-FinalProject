import pytest
import sys
import ADG4.ad as ad 
import numpy as np
import random as random

def test_basic():
    #a = 3.0 # Value to evaluate at
    a = np.random.rand(1)
    #a.type()
    #print(a)
    #a=5.0
    x = ad.AutoDiffVector(a)
    
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
    f = x/alpha
    assert f.val==a/alpha and f.der==1/alpha
    f = alpha/x
    assert f.val==alpha/a and f.der==-alpha/a**2
 
##########
###Boer Nov17
def test_trig():
    a = random.random()
    print(a)
    x=ad.AutoDiffVector(a)
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
    x=ad.AutoDiffVector(a)
    f=ad.exp_ad(x)
    assert f.val==np.exp(a) and f.der==np.exp(a)
###########
###Boer Nov17
def test_pow():
    #a = 2.0 # Value to evaluate at    
    a = np.random.rand(1)
    x = ad.AutoDiffVector(a)
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
    x = ad.AutoDiffVector(a)
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
    f5=f1/f2
    print(f5.val,f5.der)
    assert f5.val==2/3 and f5.der==0

def test_vector():
    [x,y,z,t]=ad.gen_vars([3.,np.pi,5.,3.4])
    f  = ad.AutoDiffVector.vconvert([(x + y**z)/t, ad.sin_ad(x+ad.cos_ad(100*y**3)-z**t)])
    print(f.val,f.der,f.partial(t))

    [x,y,z,t]=ad.gen_vars([3.,np.pi,5.,3.4])
    f = (x + y**z)/t
    print(f.val,f.der,f.partial(t))

    v_list=ad.gen_vars(np.linspace(1,1,10000))
    f = ad.mul_ad(v_list)
    print(f.val,f.der)
    assert 1==1
