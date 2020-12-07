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
        new=copy.deepcopy(self)
        new.name=None
        
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
        new.name=None
        
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
        else:
            raise KeyError('Function not directly dependent on input')
            




x=AutoDiffReverse(3, name='x')
y=AutoDiffReverse(4, name='y')
z=AutoDiffReverse(9, name='z')
m=x+y
n=m*z+x
print(n.val,n.partial(x),n.partial(y),n.partial(z))

print(n._partial)
assert n.partial(x) == 10
assert n.partial(y) == 1 * z.val
assert n.partial(z) == m.val


assert m.partial(x) == 1
assert m.partial(y) == 1
