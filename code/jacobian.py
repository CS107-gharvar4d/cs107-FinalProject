import numpy as np
import copy
import sys 
sys.setrecursionlimit(10**6) 

class AutoDiffVector():
    
    def __init__(self,a,der):
        self.val=a
        self.der=der

    @classmethod
    def vconvert(cls,v):
        return AutoDiffVector(np.array([ii.val for ii in v]),np.array([ii.der for ii in v]))

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

    def __truediv__(self,other):
        new=copy.copy(self)
        try:
            new.val=self.val/other.val
            new.der=self.der/other.val-self.val/np.power(other.val,2.)*other.der
        except AttirbuteError:
            new.val=self.val/other
            new.der=self.der/other
        return new

    def __truerdiv__(self,other):
        new=copy.copy(self)
        try:
            new.val=other.val/self.val
            new.der=other.der/self.val-other.val/np.power(self.val,2.)*self.der
        except AttributeError:
            new.val=other/self.val
            new.der=other/np.power(self.val,2.)*self.der
        return new

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

def mul_ad(x):
    return x[0] * mul_ad(x[1:]) if len(x)>1 else x[0]

def gen_vars(vvars):
    vars=[]
    nvars=len(vvars)
    for ii in range(len(vvars)):
        der=np.zeros(nvars)
        der[ii]=1
        vars.append(AutoDiffVector(vvars[ii],der))
    return vars

[x,y,z,t]=gen_vars([3.,np.pi,5.,3.4])
f = AutoDiffVector.vconvert([(x + y**z)/t, sin_ad(x+cos_ad(100*y**3)-z**t)])
print(f.val,f.der)

v_list=gen_vars(np.linspace(1,1,10000))
f = mul_ad(v_list)
print(f.val,f.der)

