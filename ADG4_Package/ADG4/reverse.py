import numpy as np
import copy
from collections import defaultdict

"""
reverse.py: Automatic Differentiation for Python with reverse mode.


See the examples below for some simple and advanced uses.

NOTES:

        ##import the reverse mode module
        import ADG4.reverse as rev
        #Here we give the users a choice to explicity give a name to the independent variables considering the implementation do not give an explicit order to the variables.
        #Our implementation naturally support vector inputs
        x = rev.AutoDiffReverse(3, name='x')
        y = rev.AutoDiffReverse(4, name='y')
        z = rev.AutoDiffReverse(9, name='z')

        
        Power function:
        
        f=x**x #calculate pow
        print(f.val,f.partial(x)) 
        
        
        Trig Function Examples:
        
        ##sin function
        f = rev.sin_rv(x)
        print(f.val, f.partial(x))
        ##print value and jacobian
        ##cos function
        f = rev.cos_rv(x)
        ##print value and jacobian
        print(f.val, f.partial(x))
        ##tan function
        f = rev.tan_rv(x)
        ##print value and jacobian
        print(f.val, f.partial(x))
        
        
        Exponential Function Example:
        
        f = rv.exp_rv(x)
        ##print value and jacobian
        print(f.val, f.partial(x))
        

"""

class AutoDiffReverse():
    """
    A reverse automatic differentiation variable class.
    """
    def __init__(self,a, name=None):
        """
        AutoDiffReverse class constructor. 
        ---------
        Inputs:
        :param a: the initial value of a variable
        :param name: the name of the variable, should be a string, it is optional.
        ---------
        """
        self.val= copy.deepcopy(a) # needed for np array reference management
        self.children = []
        self.has_backpropped = False
        self.name = name
        self._partial = {}

    def __repr__(self):
        """
        A print function for development purpose
        """
        if not self.name:
            return f'AutoDiffReverse({self.val})'
        return f'AutoDiffReverse({self.val}, name="{self.name}")'

    def __add__(self,other):
        """
        add function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffReverse instance
        """
        new=AutoDiffReverse(self.val)
        
        try:
            new.val+=other.val
            new.children=[[self,1],[other,1]]
        except AttributeError:
            new.val+=other
            new.children=[[self,1]]
        return new

    def __radd__(self,other):
        """
        reverse add function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffReverse instance
        """
        return self.__add__(other)

    def __mul__(self,other):
        """
        multiplication function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffReverse instance
        """
        new=AutoDiffReverse(self.val)
        
        try:
            new.val*=other.val
            new.children=[[self,other.val],[other,self.val]]
        except AttributeError:
            new.val*=other
            new.children=[[self,other]]
        return new

    def __rmul__(self,other):
        """
        reverse multiplication function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffReverse instance
        """
        return self.__mul__(other)
    
    def backprop(self):
        """
        back propogation function, which backprop the tree of partial derivatives formed by the chain rule
        """
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
        """
        Returns partial derivative given variable vv
        -----------
        :param vv: a AutoDiffReverse istance, the partial derivative will be calcuated with respect to vari
        -----------
        :return: return the partial derivative
        """
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
        """
        unary negative function
        ------------
        No input
        ------------
        output: A new AutoDiffVector instance
        """
        new=AutoDiffReverse(-self.val)
        new.name=None
        new.children=[[self,-1]]
        return new

    def __inv__(self):
        """
        unary invert function. Invert for a variable x is defined as 1/x for its value and derivative
        ------------
        No input
        ------------
        output: A new AutoDiffReverse instance
        """
        new=AutoDiffReverse(1/self.val)
        new.name=None
        new.children=[[self,-1/(self.val)**2]]
        return new
    
    def __sub__(self,other):
        """
        subtraction function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffReverse instance
        """   
        return self+(-other)

    def __rsub__(self,other):
        """
        reverse subtraction function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffReverse instance
        """
        return -self+other

    def __truediv__(self,other):
        """
        divide function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffReverse instance
        """
        try:
            return self*other.__inv__()
        except AttributeError:
            return self*(1/other)

    def __rtruediv__(self,other):
        """
        reverse divide function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffReverse instance
        """
        return other*self.__inv__()

    def __pow__(self,other):
        
        """
        power function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffReverse instance
        """
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
        """
         reverse power function
         ------------
         other: either a int/float, or a AutoDiffVector instance
         ------------
         output: A new AutoDiffReverse instance
        """
        new=AutoDiffReverse(self.val)
        new.name=None
        new.val=other**self.val
        new.children=[[self,other**self.val*np.log(other)]]
        return new

"""
Below is a set of elementary functions for AutoDiffReverse. The calculation of them are self-evident.
-----------
Input: If not particularly specified, should be a AutoDiffReverse instance
-----------
Return: return a new AutoDiffReverse instance after the calculation
"""
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
    """
    Input `a` should be a scaler variable such as a int or float. `a` is an arbitrary base for the calculation.
    """
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
    """
    We define logistic function as 1/(1+exp(-x))
    """
    return 1/(1+exp_rv(-x))

def sqrt_rv(x):
      return x**0.5
