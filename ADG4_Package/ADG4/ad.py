import numpy as np
import copy
import sys

sys.setrecursionlimit(10 ** 6)



"""
ad.py: Automatic Differentiation for Python with forward mode.
Reverse module can be found in the module reverse.py


See the examples below for some simple and advanced uses.

NOTES:

    #Examples for Simple Operations and printing value and derivative:
    f=2*x+3
    print(f.val,f.der)
    f=2/x-3
    print(f.val,f.der)
    f=x**x #calculate pow
    print(f.val,f.der)

    #Examples for trig functions:
    f = sin_ad(x)
    print(f.val, f.der)
    f = cos_ad(x)
    print(f.val, f.der)
    f = tan_ad(x)
    print(f.val, f.der)

    #Examples for exp function
    f = exp_ad(x)
    print(f.val, f.der)

"""

class AutoDiffVector():
    """
    A class for forward mode automatic differentiation variable.
    """
    def __init__(self, a, der=1):
        """
        AutoDiffVector class constructor. A single nominal value is supported as val.
        If no value given for "der", then the default
        is "1"
        :param a:
        :param der:
        """
        self.val = a
        self.der = der

    @classmethod
    def vconvert(cls, v):
        """
        vectorize the output of the function from Rm to Rn
        ---------------
        v: a list of AutoDiffVector instances
        """
        vvector=np.array([ii.val for ii in v])
        vvector=vvector.reshape(len(vvector),1)
        return AutoDiffVector(vvector, np.array([ii.der for ii in v]))

    def __add__(self, other):
        """
        add function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffVector instance
        """
         
        new = copy.deepcopy(self)
        try:
            new.val += other.val
            new.der += other.der
        except AttributeError:
            new.val += other
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        """
        mutiplication function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffVector instance
        """
        new = copy.deepcopy(self)
        try:
            new.val *= other.val
            new.der = self.der * other.val + self.val * other.der
        except AttributeError:
            new.val *= other
            new.der *= other
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        divide function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffVector instance
        """
        new = copy.deepcopy(self)
        try:
            new.val = self.val / other.val
            new.der = self.der / other.val - self.val / np.power(other.val, 2.) * other.der
        except AttributeError:
            new.val = self.val / other
            new.der = self.der / other
        return new

    def __rtruediv__(self, other):
        """
        reverse divide function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffVector instance
        """
        new = copy.deepcopy(self)
        try:
            new.val = other.val / self.val
            new.der = other.der / self.val - other.val / np.power(self.val, 2.) * self.der
        except AttributeError:
            new.val = other / self.val
            new.der = -other / np.power(self.val, 2.) * self.der
        return new

    def __neg__(self):
        """
        unary negative function
        ------------
        No input
        ------------
        output: A new AutoDiffVector instance
        """
        new = copy.deepcopy(self)
        new.val = -new.val
        new.der = -new.der
        return new

    def __sub__(self, other):
        """
        subtraction function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffVector instance
        """
        new = copy.deepcopy(self)
        try:
            new.val -= other.val
            new.der -= other.der
        except AttributeError:
            new.val -= other
        return new

    def __rsub__(self, other):
        """
        reverse subtraction function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffVector instance
        """
        return -self.__sub__(other)

    def __pow__(self, other):
        """
        power function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffVector instance
        """
        new = copy.deepcopy(self)
        try:
            new.val = np.power(self.val, other.val)
            new.der = other.val * np.power(self.val, other.val - 1) * self.der + new.val * np.log(self.val) * other.der
        except AttributeError:
            new.val = np.power(self.val, other)
            new.der = other * np.power(self.val, other - 1) * self.der
        return new

    def __rpow__(self, other):
        """
        reverse power function
        ------------
        other: either a int/float, or a AutoDiffVector instance
        ------------
        output: A new AutoDiffVector instance
        """
        new = copy.deepcopy(self)
        try:
            new.val = np.power(other.val, self.val)
            new.der = self.val * np.power(other.val, self.val - 1) * other.der + new.val * np.log(other.val) * self.der
        except AttributeError:
            new.val = np.power(other, self.val)
            new.der = new.val * np.log(other) * self.der
        return new

    def partial(self, vari):
        """
        Returns partial derivative given variab
        -----------
        :param vari: a AutoDiffVector istance, the partial derivative will be calcuated with respect to vari
        -----------
        :return: return the partial derivative
        """
        try:
            idx = np.nonzero(vari.der)[0]
            if len(idx) > 1:
                print('Not an independent variable')
                raise TypeError
            if len(self.der.shape) == 1:
                self.der = self.der.reshape(1, -1)
            return self.der[:, idx[0]]
        except AttributeError:
            print('Not an independent variable')
            raise TypeError

    # Boer Dec 5
    
    def __eq__(self, other):
        """
        compare two AutoDiffVectors
        ------------
        other: a AutoDiffVector instance
        ------------
        output: return a boolean variable, true if two instances are equal
        """
        try:
            return np.abs(self.val - other.val) < 1e-6 and np.abs(self.der - other.der) < 1e-6
        except ValueError:
            return (np.abs(self.val - other.val)).all() < 1e-6 and (np.abs(self.der - other.der)).all() < 1e-6
    def __ne__(self, other):
        """
        compare two AutoDiffVectors
        ------------
        other: a AutoDiffVector instance
        ------------
        output: return a boolean variable, true if two instances are NOT equal
        """
        return ~self.__eq__(other)


"""
Below is a set of elementary functions for AutoDiffVectors. The calculation of them are self-evident.
-----------
Input: If not particularly specified, should be a AutoDiffVector instance
-----------
Return: return a new AutoDiffVector instance after the calculation

"""

def sin_ad(x):
    y = copy.deepcopy(x)
    y.val = np.sin(x.val)
    y.der = np.cos(x.val) * x.der
    return y


def cos_ad(x):
    y = copy.deepcopy(x)
    y.val = np.cos(x.val)
    y.der = -np.sin(x.val) * x.der
    return y


def tan_ad(x):
    y = copy.deepcopy(x)
    y.val = np.tan(x.val)
    y.der = np.power(1. / np.cos(x.val), 2.) * x.der
    return y


# Boer Dec4
def arcsin_ad(x):
    y = copy.deepcopy(x)
    y.val = np.arcsin(x.val)
    y.der = 1 / (1 - x.val ** 2) ** 0.5 * x.der
    return y


def arccos_ad(x):
    y = copy.deepcopy(x)
    y.val = np.arccos(x.val)
    y.der = -1 / (1 - x.val ** 2) ** 0.5 * x.der
    return y


def arctan_ad(x):
    y = copy.deepcopy(x)
    y.val = np.arctan(x.val)
    y.der = 1 / (1 + x.val ** 2) * x.der
    return y


def expa_ad(a, x):
    """
    Input `a` should be a scaler variable such as a int or float. `a` is an arbitrary base for the calculation.

    """
    y = copy.deepcopy(x)
    y.val = a ** x.val
    y.der = a ** x.val * np.log(a) * x.der
    return y


def loga_ad(a, x):
    """
    Input `a` should be a scaler variable such as a int or float. `a` is an arbitrary base for the calculation.

    """
    y = copy.deepcopy(x)
    y.val = np.log(x.val) / np.log(a)
    y.der = 1 / (x.val * np.log(a)) * x.der
    return y


def log_ad(x):
    y = copy.deepcopy(x)
    y.val = np.log(x.val)
    y.der = 1 / (x.val) * x.der
    return y


def sinh_ad(x):
    y = copy.deepcopy(x)
    y.val = np.sinh(x.val)
    y.der = np.cosh(x.val) * x.der
    return y


def cosh_ad(x):
    y = copy.deepcopy(x)
    y.val = np.cosh(x.val)
    y.der = np.sinh(x.val) * x.der
    return y


def tanh_ad(x):
    y = copy.deepcopy(x)
    y.val = np.tanh(x.val)
    y.der = (np.cosh(x.val) ** 2 - np.sinh(x.val) ** 2) / (np.cosh(x.val) ** 2) * x.der
    return y


def logistic_ad(x):
    """
    We choose logistic function as 1/(1+exp(-x))

    """
    y = copy.deepcopy(x)
    y.val = 1 / (1 + np.exp(-x.val))
    y.der = np.exp(x.val) / (1 + np.exp(x.val)) ** 2 * x.der
    return y


def sqrt_ad(x):
    return x ** 0.5


"""
Above is a set of elementary functions for AutoDiffVectors. The calculation of them are self-evident.
-----------
Input: If not particularly specified, should be a AutoDiffVector instance
-----------
Return: return a new AutoDiffVector instance after the calculation

"""

# Boer Dec 5
def exp_ad(x):
    y = copy.deepcopy(x)
    y.val = np.exp(x.val)
    y.der = np.exp(x.val) * x.der
    return y



def mul_ad(x):
    """
    A function to mupltiply multiple AutoDiffVector variables. This is to make calculation like f=x1x2x3...x100 more convenient.
    ---------------
    vx: AutoDiffVector variables
    ---------------
    return: The result of Mutiplication
    ---------------
    """
    return x[0] * mul_ad(x[1:]) if len(x) > 1 else x[0]



def gen_vars(vvars):
    """
    vectorize the inputs of the function from Rm to Rn
    ---------------
    vvars: a list of initial values of different variables
    ---------------
    return: a list of AutoDiffVectors, which are different variables. The variables for a same function should be defined together using gen_vars.
    ---------------
    Example:
    [x,y,z,t]=ad.gen_vars([3.,np.pi,5.,3.4])
    """
    vars = []
    nvars = len(vvars)
    for ii in range(len(vvars)):
        der = np.zeros(nvars)
        der[ii] = 1
        vars.append(AutoDiffVector(vvars[ii], der))
    return vars
