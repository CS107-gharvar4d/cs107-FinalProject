import numpy as np
import copy

class AutoDiffToy():
    
    def __init__(self,a):
        self.val=a
        self.der=1

    def __add__(self,other):
        new=copy.copy(self)
        try:
            new.val+=other.val
            new.der+=other.der
        except AttributeError:
            new.val+=other
        return new

    def __radd__(self,other):
        return self.__add__(other)

    def __mul__(self,other):
        new=copy.copy(self)
        try:
            new.val*=other.val
            new.der=self.der*other.val+self.val*other.der
        except AttributeError:
            new.val*=other
            new.der*=other
        return new

    def __rmul__(self,other):
        return self.__mul__(other)

    def __neg__(self):
        new=copy.copy(self)
        new.val=-new.val
        new.der=-new.der
        return new

    def __sub__(self,other):
        new=copy.copy(self)
        try:
            new.val-=other.val
            new.der-=other.der
        except AttributeError:
            new.val-=other
        return new

    def __rsub__(self,other):
        return -self.__sub__(other)

    def __pow__(self,other):
        new=copy.copy(self)
        try:
            new.val=np.power(self.val,other.val)
            new.der=other.val*np.power(self.val,other.val-1)*self.der+new.val*np.log(self.val)*other.der
        except AttributeError:
            new.val=np.power(self.val,other)
            new.der=other*np.power(self.val,other-1)*self.der
        return new

    def __rpow__(self,other):
        new=copy.copy(self)
        try:
            new.val=np.power(other.val,self.val)
            new.der=self.val*np.power(other.val,self.val-1)*other.der+new.val*np.log(other.val)*self.der
        except AttributeError:
            new.val=np.power(other,self.val)
            new.der=new.val*np.log(other)*self.der
        return new

# David, 11/08/2020, trying to find a duner method for sin...
#    def __sin__(self):
#        new=copy.copy(self)
#        new.val=np.sin(self.val)
#        new.der=np.cos(self.val)*self.der
#        return new

def sin_ad(x):
    y=copy.copy(x)
    y.val=np.sin(x.val)
    y.der=np.cos(x.val)*x.der
    return y

def cos_ad(x):
    y=copy.copy(x)
    y.val=np.cos(x.val)
    y.der=-np.sin(x.val)*x.der
    return y

def tan_ad(x):
    y=copy.copy(x)
    y.val=np.tan(x.val)
    y.der=np.power(np.sec(x.val),2.)*x.der
    return y

def exp_ad(x):
    y=copy.copy(x)
    y.val=np.exp(x.val)
    y.der=np.exp(x.val)*x.der
    return y

a = 2.0 # Value to evaluate at
x = AutoDiffToy(a)

alpha = 2.0
beta = 3.0

f = alpha * x + beta
print(f.val, f.der)
f = x * alpha + beta
print(f.val, f.der)
f = beta + alpha * x
print(f.val, f.der)
f = beta + x * alpha
print(f.val, f.der)
f = beta - x * alpha
print(f.val, f.der)
f = sin_ad(x)
print(f.val, f.der)
f = 2**x
print(f.val, f.der)
f = x**2
print(f.val, f.der)
f = x**x
print(f.val, f.der)
f = -2**x
print(f.val, f.der)
