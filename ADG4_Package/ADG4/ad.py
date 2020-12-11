"""
ADG4: Automatic Differentiation for Python

A simple tool for handling automatic differentiation (AD) in Python using
elementary functions, as well as, trigonometry
See the examples at the bottom for some simple and advanced uses.
NOTES:

    #Examples for Simple Operations and printing value and derivative:
    f=alpha*x+beta
    print(f.val,f.der)
    f=alpha/x-beta
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

import numpy as np
import copy
import sys

sys.setrecursionlimit(10 ** 6)


class AutoDiffVector():

    def __init__(self, a, der=1):
        """
        AD class constructor. A single nominal value is supported as val.
        If no value given for "der", then the default
        is "1"
        :param a:
        :param der:
        """
        self.val = a
        self.der = der

    @classmethod
    def vconvert(cls, v):
        vvector=np.array([ii.val for ii in v])
        vvector=vvector.reshape(len(vvector),1)
        return AutoDiffVector(vvector, np.array([ii.der for ii in v]))

    def __add__(self, other):
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
        new = copy.deepcopy(self)
        try:
            new.val = self.val / other.val
            new.der = self.der / other.val - self.val / np.power(other.val, 2.) * other.der
        except AttributeError:
            new.val = self.val / other
            new.der = self.der / other
        return new

    def __rtruediv__(self, other):
        new = copy.deepcopy(self)
        try:
            new.val = other.val / self.val
            new.der = other.der / self.val - other.val / np.power(self.val, 2.) * self.der
        except AttributeError:
            new.val = other / self.val
            new.der = -other / np.power(self.val, 2.) * self.der
        return new

    def __neg__(self):
        new = copy.deepcopy(self)
        new.val = -new.val
        new.der = -new.der
        return new

    def __sub__(self, other):
        new = copy.deepcopy(self)
        try:
            new.val -= other.val
            new.der -= other.der
        except AttributeError:
            new.val -= other
        return new

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __pow__(self, other):
        new = copy.deepcopy(self)
        try:
            new.val = np.power(self.val, other.val)
            new.der = other.val * np.power(self.val, other.val - 1) * self.der + new.val * np.log(self.val) * other.der
        except AttributeError:
            new.val = np.power(self.val, other)
            new.der = other * np.power(self.val, other - 1) * self.der
        return new

    def __rpow__(self, other):
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
        Returns partial derivative given variable
        :param vari:
        :return:
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
        try:
            return np.abs(self.val - other.val) < 1e-6 and np.abs(self.der - other.der) < 1e-6
        except ValueError:
            return (np.abs(self.val - other.val)).all() < 1e-6 and (np.abs(self.der - other.der)).all() < 1e-6
    def __ne__(self, other):
        return ~self.__eq__(other)


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
    y = copy.deepcopy(x)
    y.val = a ** x.val
    y.der = a ** x.val * np.log(a) * x.der
    return y


def loga_ad(a, x):
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
    y = copy.deepcopy(x)
    y.val = 1 / (1 + np.exp(-x.val))
    y.der = np.exp(x.val) / (1 + np.exp(x.val)) ** 2 * x.der
    return y


def sqrt_ad(x):
    return x ** 0.5


# Boer Dec 5
def exp_ad(x):
    y = copy.deepcopy(x)
    y.val = np.exp(x.val)
    y.der = np.exp(x.val) * x.der
    return y


def mul_ad(x):
    return x[0] * mul_ad(x[1:]) if len(x) > 1 else x[0]


def gen_vars(vvars):
    vars = []
    nvars = len(vvars)
    for ii in range(len(vvars)):
        der = np.zeros(nvars)
        der[ii] = 1
        vars.append(AutoDiffVector(vvars[ii], der))
    return vars
