import numpy as np
import copy

class AutoDiffToy():
    
    def __init__(self,a):
        self.val=copy.deepcopy(a) ##Boer Nov.17 added a "copy.copy" to address a case where a is initiated by a (1,) ndarray
        self.der=1

    def __add__(self,other):
        new=copy.deepcopy(self) ##Boer change it to deepcopy to address a case where the original x is changed
        try:
            new.val+=other.val
            new.der+=other.der
        except AttributeError:
            new.val+=other
        return new

    def __radd__(self,other):
        return self.__add__(other)

    def __mul__(self,other):
        new=copy.deepcopy(self)
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
        new=copy.deepcopy(self) #Boer change it to deepcopy to avoid self to be changed Nov.17
        new.val=-new.val
        new.der=-new.der
        return new

    def __sub__(self,other):
        new=copy.deepcopy(self) #Boer change it to deepcopy to avoid self to be changed Nov.17
        try:
            new.val-=other.val
            new.der-=other.der
        except AttributeError:
            new.val-=other
        return new

    def __rsub__(self,other):
        return -self.__sub__(other)

    def __pow__(self,other):
        new=copy.deepcopy(self) #Boer change it to deepcopy to avoid self to be changed Nov.17
        try:
            new.val=np.power(self.val,other.val)
            new.der=other.val*np.power(self.val,other.val-1)*self.der+new.val*np.log(self.val)*other.der
        except AttributeError:
            new.val=np.power(self.val,other)
            new.der=other*np.power(self.val,other-1)*self.der
        return new

    def __rpow__(self,other):
        new=copy.deepcopy(self) #Boer change it to deepcopy to avoid self to be changed Nov.17
        try:
            new.val=np.power(other.val,self.val)
            new.der=self.val*np.power(other.val,self.val-1)*other.der+new.val*np.log(other.val)*self.der
        except AttributeError:
            new.val=np.power(other,self.val)
            new.der=new.val*np.log(other)*self.der
        return new

# David, 11/08/2020, trying to find a dunder method for sin...
#    def __sin__(self):
#        new=copy.deepcopy(self) #Boer change it to deepcopy to avoid self to be changed Nov.17
#        new.val=np.sin(self.val)
#        new.der=np.cos(self.val)*self.der
#        return new

def sin_ad(x):
    y=copy.deepcopy(x) #Boer change it to deepcopy to avoid self to be changed Nov.17
    y.val=np.sin(x.val)
    y.der=np.cos(x.val)*x.der
    return y

def cos_ad(x):
    y=copy.deepcopy(x) #Boer change it to deepcopy to avoid self to be changed Nov.17
    y.val=np.cos(x.val)
    y.der=-np.sin(x.val)*x.der
    return y

def tan_ad(x):
    y=copy.deepcopy(x) #Boer change it to deepcopy to avoid self to be changed Nov.17
    y.val=np.tan(x.val)
    y.der=np.power(1/np.cos(x.val),2.)*x.der ##Boer Nov 17 2020 Looks like np do not have sec,so change it to 1/cos(x.val)
    return y

def exp_ad(x):
    y=copy.deepcopy(x) #Boer change it to deepcopy to avoid self to be changed Nov.17
    y.val=np.exp(x.val)
    y.der=np.exp(x.val)*x.der
    return y

a = 2.0 # Value to evaluate at
x = AutoDiffToy(a)

alpha = 2.0
beta = 3.0

#f = alpha * x + beta
#print(f.val, f.der)
#f = x * alpha + beta
#print(f.val, f.der)
#f = beta + alpha * x
#print(f.val, f.der)
#f = beta + x * alpha
#print(f.val, f.der)
f = beta - x * alpha
print(f.val, f.der)
#f = sin_ad(x)
#print(f.val, f.der)
#f = 2**x
#print(f.val, f.der)
#f = x**2
#print(f.val, f.der)
#f = x**x
#print(f.val, f.der)
#f = -2**x
#print(f.val, f.der)
