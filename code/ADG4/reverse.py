import numpy as np
import copy



class AutoDiffReverse():
    
    def __init__(self,a):
        self.val=a
        self.derivs = { self: 0}
        self.children = []
        self.acc = 1


    def get_deriv(self,wrt=None):
        if wrt:
            return self.derivs[wrt]
        return self.derivs


    def __repr__(self):
        return f'AutoDiffToy({self.val})'

    def __add__(self,other):
        new=copy.deepcopy(self)
        
        try:
            new.val+=other.val
            new.children=[[self,1],[other,1]]
        except AttributeError:
            new.val+=other
            new.children=[(self,1)]
        return new

    def __radd__(self,other):
        return self.__add__(other)

    def __mul__(self,other):
        new=copy.deepcopy(self)
        try:
            new.val*=other.val
            new.children=[[self,other.val],[other,self.val]]
        except AttributeError:
            new.val*=other
            new.children=[[self,other]]
        return new

    def __rmul__(self,other):
        return self.__mul__(other)
    
    def backprop(self):
        var_list=[]
        self._backprop(var_list)
        self._partial={}
        for vv in var_list:
            self._partial[vv]=vv.derivs[vv]

    def _backprop(self,var_list):
        for (c,i) in self.children:
            c.acc = i * self.acc
            c._backprop(var_list)
        if not self.children:
            self.derivs[self]=self.acc+self.derivs[self]
            var_list.append(self)

    def partial(self,vv):
        self.backprop()
        if vv in self._partial.keys():
            return self._partial[vv]
        else:
            raise KeyError('Function not directly dependent on input')
            




x=AutoDiffReverse(3)
y=AutoDiffReverse(4)
z=AutoDiffReverse(9)
m=x+y
n=m*z+x
print(n.val,n.partial(x))
