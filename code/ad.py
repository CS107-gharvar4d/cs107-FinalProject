import numpy as np
import copy



class AutoDiffToy():
    
    def __init__(self,a):
        self.val=a
        self.derivs = { self: 1}


    def get_deriv(wrt=None):
        if wrt:
            return self.derivs[wrt]
        return self.derivs

    def merge_derivs(self, d1, d2, fn=lambda deriv_1, deriv_2: deriv_1 + deriv_2, default_value=0):
        res = {}

        for var, d1_deriv in d1.items():
            if var in d2:
                res[var] = fn(d1_deriv,d2[var])
            else:
                res[var] = fn(d1_deriv,default_value)

        for var, d2_deriv in d2.items():
            if var not in res:
                res[var] = fn(default_value,d2_deriv)
        return res



    def map_derivs(self, fn=lambda deriv: deriv):
        res = self.derivs.copy()
        for key, val in res.items():
            res[key] = fn(val)
        return res


    def __repr__(self):
        return f'AutoDiffToy({self.val})'

    def __add__(self,other):
        new=copy.copy(self)
        try:
            new.val+=other.val
            new.derivs = self.merge_derivs(new.derivs, other.derivs)
        except AttributeError:
            new.val+=other
        return new

    def __radd__(self,other):
        return self.__add__(other)

    def __mul__(self,other):
        new=copy.copy(self)
        try:
            new.val*=other.val
            new.derivs = self.merge_derivs(new.derivs, other.derivs, lambda new_der, other_der: new_der*other.val + self.val * other_der)
        except AttributeError:
            new.val*=other
            new.derivs = new.map_derivs(lambda deriv: deriv*other)
        return new

    def __rmul__(self,other):
        return self.__mul__(other)

    def __neg__(self):
        new=copy.copy(self)
        new.val=-new.val
        new.derivs = new.map_derivs(lambda deriv: -deriv)
        return new

    def __sub__(self,other):
        new=copy.copy(self)
        try:
            new.val-=other.val
            new.derivs = self.merge_derivs(new.derivs, other.derivs, lambda new_der,other_der: new_der - other_der)

        except AttributeError:
            new.val-=other
        return new

    def __rsub__(self,other):
        return -self.__sub__(other)

    def __pow__(self,other):
        new=copy.copy(self)
        try:
            new.val=np.power(self.val,other.val)
            l = lambda new_der,other_der: other.val*np.power(self.val,other.val-1)*new_der+new.val*np.log(self.val)*other_der
            new.derivs = self.merge_derivs(new.derivs, other.derivs, l)
        except AttributeError:
            new.val=np.power(self.val,other)
            new.derivs= new.map_derivs(lambda deriv: other*np.power(self.val,other-1)*deriv)
        return new

    def __rpow__(self,other):
        new=copy.copy(self)
        try:
            new.val=np.power(other.val,self.val)
            l = lambda new_der,other_der: self.val*np.power(other.val,self.val-1)*other_der+new.val*np.log(other.val)*new_der
            new.derivs = self.merge_derivs(new.derivs, other.derivs, l)
        except AttributeError:
            new.val=np.power(other,self.val)
            new.derivs= new.map_derivs(lambda deriv: new.val*np.log(other)*deriv)
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
    y.derivs = x.map_derivs(lambda x_der: np.cos(x.val)*x_der)

    return y

def cos_ad(x):
    y=copy.copy(x)
    y.val=np.cos(x.val)
    y.derivs =  x.map_derivs(lambda x_der: -np.sin(x.val)*x_der)
    return y

def tan_ad(x):
    y=copy.copy(x)
    y.val=np.tan(x.val)
    y.derivs =  x.map_derivs(lambda x_der: np.power(np.sec(x.val),2.)*x_der)
    return y

def exp_ad(x):
    y=copy.copy(x)
    y.val=np.exp(x.val)
    y.derivs =  x.map_derivs(lambda x_der: np.exp(x.val)*x_der)
    return y
