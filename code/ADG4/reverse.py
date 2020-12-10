import numpy as np
import copy
from collections import defaultdict



class AutoDiffReverse():
    
    def __init__(self,a, name=None):
        self.val=a
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

x=AutoDiffReverse(3, name='x')
y=AutoDiffReverse(4, name='y')
z=AutoDiffReverse(9, name='z')
m=x*y
print("m",m)
m=x+y
n=m*z+x

p=x*x*x

print(n.val,n.partial(x),n.partial(y),n.partial(z))

print(n._partial)
assert n.partial(x) == 10
assert n.partial(y) == 1 * z.val
assert n.partial(z) == m.val


assert m.partial(x) == 1
assert m.partial(y) == 1

assert p.partial(x)==3*3*3


##test_neg
q=-n
assert q.partial(x)==-n.partial(x)

##test_sub
m=x-y
assert m.val==-1
assert m.partial(x)==1
assert m.partial(y)==-1
m=x-1
assert m.val==2
assert m.partial(x)==1
m=1-x
assert m.val==-2
assert m.partial(x)==-1


##test_inv
r=x.__inv__()
assert  r.partial(x)==-1/x.val**2
##test_div
m=x/y
assert m.val==3/4
assert m.partial(x)==1/4
assert m.partial(y)==3*-(4)**(-2)
m=x/2

assert m.val==1.5
assert m.partial(x)==1/2
m=2/x
assert m.val==2/3
assert m.partial(x)==2*-(3)**(-2)

##test pow
m=x**y
assert m.val==81
assert m.partial(x)==4*3*3*3
assert m.partial(y)==3**4*np.log(3)

m=2**x
assert m.val==8
assert m.partial(x)==2**3*np.log(2)
