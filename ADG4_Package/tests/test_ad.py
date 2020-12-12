"""
A test file for the forward mode ad.py. The name of the tests are self-evident. 
"""
import pytest
import sys
import ADG4.ad as ad 
import numpy as np
import random as random
import pandas as pandas

##Boer Dec5, calculating derivative of f at x0  using finite difference
def fprime_fd(f,x0,dx=1e-12):
    return (f(x0+dx)-f(x0))/dx

##test for basic calculations
def test_basic():
    a = np.random.rand(1)
    x = ad.AutoDiffVector(a)
    
    alpha = 2.0
    beta = 3.0

    f =  alpha*x+beta
    
    assert f.val==alpha*a+beta and f.der==alpha
    f = x * alpha + beta
    assert f.val==alpha*a+beta and f.der==alpha
    f = beta + alpha * x
    assert f.val==alpha*a+beta and f.der==alpha
    f = beta + x * alpha
    assert f.val==alpha*a+beta and f.der==alpha 
    f = beta - x * alpha
    assert f.val==beta-a*alpha and f.der==-alpha
    f = x/alpha
    assert f.val==a/alpha and f.der==1/alpha
    f = alpha/x
    assert f.val==alpha/a and f.der==-alpha/a**2
 
def test_not_ind_variable():
    a = 2.0
    x = ad.AutoDiffVector(a)
    alpha = 2.0
    beta = 3.0
    f = alpha * x + beta
    m = f * x
    with pytest.raises(Exception):
        m.partial(f)
        
###Boer Nov17, test for elementary functions
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
    x=ad.AutoDiffVector(a)
    f=ad.exp_ad(x)
    assert f.val==np.exp(a) and f.der==np.exp(a)

###Boer Nov17
def test_pow(): 
    a = np.random.rand(1)
    x = ad.AutoDiffVector(a)
    f1= 2**x
    assert f1.val==2**a and np.abs(f1.der-2**a*np.log(2))<1e-5    
    f2= x**2
    assert f2.val==a**2 and f2.der==2*a
    f3= x**x
    assert f3.val==a**a and np.abs(f3.der-(np.log(a)+1)*a**a)<1e-5
    f4= x.__rpow__(x)
    assert f4.val==a**a and np.abs(f4.der-(np.log(a)+1)*a**a)<1e-5


def test_unary_negation():
    a = 80
    x = ad.AutoDiffVector(a)
    f = -x
    assert f.val == -80
    assert f.der == -1



def test_twoAD():
    a = 2.0 
    x = ad.AutoDiffVector(a)
    f1= 2*x
    f2=f1+x
    f3=f1-x
    f4=f1*x
    assert f1.val==4.0 and f1.der==2.0
    assert f2.val==6.0 and f2.der==3
    assert f3.val==2 and f3.der==1
    assert f4.val==8 and f4.der==8
    f5=f1/f2
    assert f5.val==2/3 and f5.der==0
    f5=f1.__rtruediv__(f2)
    assert f5.val==3/2 and f5.der==0

def test_vector():

    [x,y,z,t]=ad.gen_vars([3.,np.pi,5.,3.4])
    f  = ad.AutoDiffVector.vconvert([(x + y**z)/t, ad.sin_ad(x+ad.cos_ad(100*y**3)-z**t)])
    
    with pytest.raises(TypeError):
        f.partial([x,y])
    [xval,yval,zval,tval]=[3.,np.pi,5.,3.4]
    assert f.val[0]==(xval + yval**zval)/tval
    assert f.val[1]==np.sin(xval+np.cos(100*yval**3)-zval**tval)
    fder= [np.array([2.94117647e-01,1.43248663e+02,1.03032317e+02,-2.67318066e+01]),np.array([-9.46178240e-01,3.47280451e+02,1.53101654e+02,3.62364128e+02])]
    
    [x1,y1,z1,t1]=ad.gen_vars([3.,np.pi,5.,3.4])
    f2  = ad.AutoDiffVector.vconvert([(x1 + y1**z1)/t1, ad.sin_ad(x1+ad.cos_ad(100*y1**3)-z1**t1)])
    print(ad.sin_ad(f2)) 
    assert f==f2
    
    assert f!=x
    
    x2=ad.AutoDiffVector(1)
    assert f!=x2
    
    [x,y,z,t]=ad.gen_vars([3.,np.pi,5.,3.4])
    f = (x + y**z)/t
    assert f.partial(t)==-(xval + yval**zval)*(tval)**-2
    with pytest.raises(TypeError):
        f.partial(1)


    v_list=ad.gen_vars(np.linspace(1,1,10000))
    f = ad.mul_ad(v_list)
    assert f.val==1
    assert (f.der==np.ones(10000)).all()
    
    
    
    
    
    
    
    


def test_expa():
    a = random.random()
    x=ad.AutoDiffVector(a)
    f=ad.expa_ad(2,x)
    assert f.val==2**a and f.der==2**a*np.log(2)

def test_loga():
    a = random.random()
    x=ad.AutoDiffVector(a)
    f=ad.loga_ad(2,x)
    assert f.val==np.log(a)/np.log(2) and f.der==1/(a*np.log(2))

def test_log():
    a = random.random()
    x=ad.AutoDiffVector(a)
    f=ad.log_ad(x)
    assert f.val==np.log(a) and f.der==1/(a)

def test_inverse_trig():
    a = random.random()
    x=ad.AutoDiffVector(a)
    f=ad.arcsin_ad(x)
    assert f.val==np.arcsin(a)
    assert  np.abs(f.der-fprime_fd(np.arcsin,a))<1e-2
    f=ad.arccos_ad(x)
    assert f.val==np.arccos(a)
    assert  np.abs(f.der-fprime_fd(np.arccos,a))<1e-2
    f=ad.arctan_ad(x)
    assert f.val==np.arctan(a)
    assert  np.abs(f.der-fprime_fd(np.arctan,a))<1e-2

def test_sqrt():
    a = random.random()
    x=ad.AutoDiffVector(a)
    f=ad.sqrt_ad(x)
    assert f.val==a**0.5
    assert f.der==0.5*a**(-0.5)
     
def test_hyperbolic():
    a = random.random()
    x=ad.AutoDiffVector(a)
    f=ad.sinh_ad(x)
    assert f.val==np.sinh(a)
    assert f.der==np.cosh(a)
    f=ad.cosh_ad(x)
    assert f.val==np.cosh(a)
    assert f.der==np.sinh(a)
    f=ad.tanh_ad(x)
    assert f.val==np.tanh(a)
    assert np.abs(f.der-(1-np.tanh(a)**2))<1e-3

def test_logistic():
    a = random.random()
    x=ad.AutoDiffVector(a)
    f=ad.logistic_ad(x)
    assert f.val==1/(1+np.exp(-a))
    assert np.abs(f.der-fprime_fd(lambda x:1/(1+np.exp(-x)),a))<1e-2

##test compare functions
def test_compare():
    a = random.random()
    x=ad.AutoDiffVector(a)
    f1=ad.sinh_ad(x)
    f2=ad.cosh_ad(x)
    f3=ad.tanh_ad(x)
    assert f1!=f2
    f4=f1/f2
    print(f4.val,f4.der)
    print(f3.val,f3.der)
    assert f1/f2==f3


