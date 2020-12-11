import numpy as np
import copy
from collections import defaultdict



class AutoDiffReverse():
    ## A reverse autodifferentiation class
    def __init__(self,a, name=None):
        self.val= copy.deepcopy(a) # needed for np array reference management
        self.children = []
        self.acc = 1
        self.has_backpropped = False
        self.name = name
        self._partial = {}

    def __repr__(self):
        if not self.name:
            return f'AutoDiffReverse({self.val})'
        return f'AutoDiffReverse({self.val}, name="{self.name}")'

    def __add__(self,other):
        new=AutoDiffReverse(self.val)
        
        try:
            new.val+=other.val
            new.children=[[self,1],[other,1]]
        except AttributeError:
            new.val+=other
            new.children=[[self,1]]
        return new

    def __radd__(self,other):
        return self.__add__(other)

    def __mul__(self,other):
        new=AutoDiffReverse(self.val)
        
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
        # A back prop implementation that keeps all derivative accumulations
        # within this root node that calls .backprop()
        partial = defaultdict(int)
        
        stack = [ (child_node, edge_value) for (child_node, edge_value) in self.children]
        while stack:
            # edge value is the derivative between the root node
            # and this current node
            node, edge_value = stack.pop()
            # Update the partial derivative to add this current derivative value
            partial[node] += edge_value
            # Add each child to our stack
            for child_node, child_edge_value in node.children:
                # For each child, its derivative with respect to the root is
                # (derivative root -> node) * (derivative node -> child_node)
                stack.append((child_node, edge_value * child_edge_value))
                
        self._partial = dict(partial)
        

    def partial(self,vv):
        if not self.has_backpropped:
            self.backprop()
            self.has_backpropped = True
        
        if vv in self._partial.keys():
            return self._partial[vv]
        ##Deal with the case when
        elif self is vv:
            return 1
        else:
            raise KeyError('Function not dependent on input')

    def __neg__(self):
      new=AutoDiffReverse(-self.val)
      new.name=None
      new.children=[[self,-1]]
      return new

    def __inv__(self):
      new=AutoDiffReverse(1/self.val)
      new.name=None
      new.children=[[self,-1/(self.val)**2]]
      return new
    
    def __sub__(self,other):
      return self+(-other)

    def __rsub__(self,other):
      return -self+other

    def __truediv__(self,other):
      try:
        return self*other.__inv__()
      except AttributeError:
        return self*(1/other)

    def __rtruediv__(self,other):
      return other*self.__inv__()

    def __pow__(self,other):
      new=AutoDiffReverse(self.val)
      new.name=None
      try:
        new.val**=other.val
        new.children=[[self,other.val*self.val**(other.val-1)],[other,self.val**other.val*np.log(self.val)]]
      except:
        new.val**=other
        new.children=[[self,other*self.val**(other-1)]]
      return new

    def __rpow__(self,other):
      new=AutoDiffReverse(self.val)
      new.name=None
      new.val=other**self.val
      new.children=[[self,other**self.val*np.log(other)]]
      return new


def sin_rv(x):
  new=AutoDiffReverse(np.sin(x.val))
  new.name=None
  new.children=[[x,np.cos(x.val)]]
  return new

def cos_rv(x):
  new=AutoDiffReverse(np.cos(x.val))
  new.name=None
  new.children=[[x,-np.sin(x.val)]]
  return new

def tan_rv(x):
  return sin_rv(x)/cos_rv(x)

def arcsin_rv(x):
  new=AutoDiffReverse(np.arcsin(x.val))
  new.name=None
  new.children=[[x,1 / (1 - x.val ** 2) ** 0.5]]
  return new

def arccos_rv(x):
  new=AutoDiffReverse(np.arccos(x.val))
  new.name=None
  new.children=[[x,-1 / (1 - x.val ** 2) ** 0.5]]
  return new

def arctan_rv(x):
  new=AutoDiffReverse(np.arctan(x.val))
  new.name=None
  new.children=[[x,-1 / (1 - x.val ** 2) ** 0.5]]
  return new

def expa_rv(a,x):
  return a**x

def exp_rv(x):
  return expa_rv(np.exp(1),x)

def loga_rv(a,x):
  new=AutoDiffReverse(np.log(x.val)/np.log(a))
  new.name=None
  new.children=[[x,1 / (x.val * np.log(a)) ]]
  return new

def log_rv(x):
  return loga_rv(np.exp(1),x)

def sinh_rv(x):
  new=AutoDiffReverse(np.sinh(x.val))
  new.name=None
  new.children=[[x,np.cosh(x.val)]]
  return new

def cosh_rv(x):
  new=AutoDiffReverse(np.cosh(x.val))
  new.name=None
  new.children=[[x,np.sinh(x.val)]]
  return new

def tanh_rv(x):
  new=AutoDiffReverse(np.tanh(x.val))
  new.name=None
  new.children=[[x,(np.cosh(x.val) ** 2 - np.sinh(x.val) ** 2) / (np.cosh(x.val) ** 2)]]
  return new

def logistic_rv(x):
  return 1/(1+exp_rv(-x))

def sqrt_rv(x):
  return x**0.5
