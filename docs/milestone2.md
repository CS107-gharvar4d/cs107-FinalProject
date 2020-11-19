# Introduction
ADG4 is a package which computes a value while using automatic
differentiation to compute derivatives of that value. It integrates a set of techniques
to evaluate the derivative of various mathematical expressions.

Being able to take derivatives has many important practical applications across a variety of domains.
In physics, the first order derivative tells us the rate of change, while further orders can tell us
more pieces of information like acceleration and actergy. In statistics, applications include Bayesian
inference and the training of neural networks. In economics, taking the derivative of profit and utility
functions allows agents to maximize their expected outcomes. These are only a few instances wherein 
differentiating complex functions is useful and necessary.

Given the broad set of applications across domains, we hope ADG4 can be a useful tool
that can facilitate quick and easy numerical evaluation of derivatives through computing.


# Background


The methodology behind our package relies on the fact that every function can be computed as
a combination of simple elementary functions, which have simple derivatives (exp, log, sin, cos, etc.).
By applying the chain rule repeatedly to these operations, derivatives (or more generally, 
Jacobian matries) of arbitrary functions can automatically calculated. 

The basic functions of our package are based on the forward mode of auto differentiation.
The forward mode takes the first step as putting in seeding variables at the place of each
independent variable, and then recursively move forward following the chain rule to 
combine all the sub-expressions to get the final derivatives. In other words, forward mode
involves repeatedly substituting the derivatives of the inner functions in the chain rule.
We are going to leverage dual numbers to operate the forward mode automatic differentiation, 
which are vectors containing a function value and derivative. The algebra of dual numbers is 
exactly analogous to the forward accumulation process of calculating the derivatives following
the chain rule.


As opposed to forward mode, reverse mode is another way of calculating automatic differentiation,
which can be a helpful extension of our package. Reverse method takes the first step as
calculating the derivatives of the outermost functions, and then using the chain rule to 
recursively calculate the derivatives of the inner functions, until it gets to each 
independent variables, where the seeding variables would be put in.

Here are a few important concepts pertaining AD which are mentioned above:

The Chain Rule: the derivative of a convoluted function is the product of each simple function evaluated at the value of its child function.
Jacobian: the gradient of each element of a function's output with respect to each and every input. In other words, it's the gradient of the function at the space spanned by the inputs.
Dual numbers: a two dimentional space where a outer product is defined between any vectors $$x\ =\ (a,\ b)x = (a, b)$$ and $$y\ =\ (c,\ d)y = (c, d)$$ as $$x\times y\ =\ (a\cdot c,\ a\cdot d\ +\ c\cdot b)x×y = (a⋅c, a⋅d + c⋅b).$$ Dual number is useful when we want to include a second-order calculation, i.e., not only calculate the value of a function, but also its changes with regard to small changes from the inputs.


# How to use

Here are the steps to use our package. First, below are the steps to download and install `ADG4`:
1. Create and activte a virtual environment, either with conda 
```
conda create -n adg4_env python=3.8
conda activate adg4_env 
```
 or any prefered method
 
2. Download our repository, `git clone git@github.com:CS107-gharvar4d/cs107-FinalProject.git`
3. Navigate into the repo folder with: `cd cs107-FinalProject`
4. Install requirements with `pip install -r requirements.txt`
5. Install our package with `pip install --editable ./code`
6. Now you can do `import ADG4.ad as ad` or just run our tests with the command `pytest` in the repo directory


### A demo

- A simple example of the user interface for how to use the package is below. Running our packages involves:
- Functional inputs: A class should be called to instantiate the object. The constructor requires the following inputs: a list of function inputs as declaration, a list of input values, a function form (methods for repetition and recursion should be provided in preparation of cases like f = x1 x2 ... x100000)
- Jacobian: Lastly, the function returns to the Jacobian matrix. 

```

import ADG4.ad as ad

a = 2.0 # Value to evaluate at
x = ad.AutoDiffVector(a) #create a AutoDiff variable with value 2
alpha=2.0
beta=3.0

###Examples for Simple Operations: Create Functions from AD Variables
f=alpha*x+beta # behind the scenes this calls the __add__ function and the __mul__ function respectively

# Now you can access the values and derivatives from the AD objects
print(f.val,f.der) # print the value and derivative

f=alpha/x-beta # behind the scenes this calls the __truediv__ function and the __sub__ function respectively
print(f.val,f.der) # print the value and derivative

f=x**x #calculate pow
print(f.val,f.der) 

##Examples for trig functions:
f = ad.sin_ad(x)
print(f.val, f.der)
f = ad.cos_ad(x)
print(f.val, f.der)
f = ad.tan_ad(x)
print(f.val, f.der)

##Examples for exp function
f = ad.exp_ad(x)
print(f.val, f.der)
```

# Software Organization

This section addresses how we plan to organize our software package.

What will the directory structure look like?
* Directory will be structured based on functionality. Modules will be deployed according to program features such as mathematic calculations, user interface, computational graph and unit tests.
* A primary influence on the directory structure is that if we want it to be pip installable, we'll need a general format that looks like this:

```
ADG4/
ad_extension/ # name TBD
tests/
docs/
setup.py
README.md
```


* We will have a unittest test file per implementation file.

```
ADG4/
	advar.py
	admath.py
ad_extension/
	extension.py
tests/
	test_advar.py
	test_admath.py
	test_extension.py
docs/
	howto.md
setup.py
README.md
```

What modules do you plan on including? What is their basic functionality?
* For now, we plan of having two models: `ADG4` for implementing our core AD functionality and `ad_extension` which will use our core library for an end-user program.
* We are also considering the use of third-party libraries or modules at this time. This will help us support specific features of the project, but nothing specific has been defined yet. But we most probably consider numpy or scipy for data structures and basic operators. We'd like to use numpy . Numpy is a mathematical computation library that makes it easy to build interactions between scalars, vectors, and matrices. It has built in support for matrix/vector math which will be useful for our final implementation and is accessible by running the following:
```
import numpy as np
```

Where will your test suite live? Will you use TravisCI? CodeCov?
* Yes we plan to use both TravisCI and CodeCov. 
* The project will leverage the unittest module to test, and will live in a separate directory structure as seen above. The test suite will be run automatically via TravisCI everytime we push a change into our branch. Each time code is pushed, they both will run all the tests in the `tests/` dir. Possibly with `python -m unittest tests/`
How will you distribute your package (e.g. PyPI)?
* We intend to distribute our package via PyPI. For the moment will do do so via git clone or forking the project, as well as, ake it friendly to use on different environments such as conda, virtualenv, etc.
* We are interested in designing the package to be in the pip package format so that it could be installed via PyPI or via `git clone` then `pip install`.
* Our main usage within our team will be with the command `pip install --editable [project repo location]`. This creates a local pip package that we can continue to live edit and interact with.

How will you package your software? Will you use a framework? If so, which one and why? If not, why not?
* We will use SetupTools (setup.py) to package our software. That way it can handle downloading dependencies and setup processes.

Other considerations?
* As noted in the project instructions we will also include a broader impact statement for our library. This will consider the accessibility of our software library to different groups of people and ensure that it is accessible and usable to a wide and representative population.

# Implementation

The core data structure is an AutoDiffVector which maintains two core pieces of data:
the current value, and the the derivatives with regard to *all* the independent variables.
Eg the partial derivatives with respect to each input variable.


## Core Classes 
### AutoDiffVector
```
AutoDiffVector
  fields:
    - val: the currently computed value
    - der: the jacobian with respect to all the input (aka. independent) variables.
  methods:
    - partial(vari)
    - __mul__, __add__, .. call the associated ADFunctions to return new AutoDiffVector
```

### Managing derivatives
At any point, the method `partial(vari=v)` can be called on an AutoDiffVector get the derivative with respont to any input variables.


## ADFunctions
ADFunctions accept one or more AutoDiffVectors and output a new AutoDiffVector with an updated `val` and `der`.
```
ADFunction
  input: One or more AutoDiffVectors
  output: A new AutoDiffVector with appropriate val, der
```

### ADFunction Implementations
Since ADFunctions do not contain internal state, and cannot be loaded as dunder methods (e.g., sin, cos), for now it seems to make the most sense to have them as direct functions.
Elementary functions like sin, cos, etc will each have a corresponding `ADFunction` implementation that stores gradients
and maintains logic depending on whether the input is a vector, matrix etc. For functions such as multiplication and addition,
which have implicit functionality in Python, we will overwrite the underlying dunder method so that when called upon an `AutoDiffVector`,
these functions will evaluate and update the derivatives as described above.

No external dependencies are required at this time.


### Scalar and Vector
A Vector is a `list(AutoDiffVectors)` 
If each `ADFunction` handles the three input types:
- Scalar
- AutoDiffVector
- Vector

We anticipate that our program will be able to support these instances.

####  Scalars, Vectors example:

```
v = AutoDiffVector(range(1)) # a vector of length 4
z = ad.sin_ad(v)

v = AutoDiffVector(range(4)) # a vector of length 4
z = ad.sin_ad(v)
```

### Example Code Structure
All functions will return a new AutoDiffVector with an updated value and an array of derivatives with respect to variables.

For example, below is an example of how we might implement a function for addition, which would perform the elementary operation of adding two AutoDiffVectors and calculating the updated derivatives.

```
    def __add__(self,other):
        new=copy.deepcopy(self)
        try:
            new.val+=other.val
            new.der+=other.der
        except AttributeError:
            new.val+=other
        return new
```

Below is another example of how we might implement a function for multiplying two scalars, again performing the elementary operation and calculating the derivatives.
```
    def __mul__(self,other):
        new=copy.deepcopy(self)
        try:
            new.val*=other.val
            new.der=self.der*other.val+self.val*other.der
        except AttributeError:
            new.val*=other
            new.der*=other
        return new
```

AutoDiffVector will track their trace within the `der` variable, just like it did with a single input case, except that `der` now is a vector. An example of this implementation can be seen below. On an AutoDiffVector, the method `partial` allows the user to extract a specific derivative of that AutoDiffVector.
```
    def __init__(self,a,der=1):
        self.val=a
        self.der=der

    def partial(self,vari):
        try:
            idx=np.nonzero(vari.der)[0]
            if len(idx)>1:
               print('Not an independent variable')
               raise TypeError
            if len(self.der.shape)==1:
               self.der=self.der.reshape(1,-1)
            return self.der[:,idx[0]]
        except AttributeError:
            print('Not an independent variable')
            raise TypeError

```
## Future Features

As an add-on feature that we've thought to include in our project is to apply AD to a a real world problem, coupled by a potential UI to facilitate this. There has been emerging conversations between Academia and the Quantitative Finance Industry of using this type of techniques for valuating derivatives. A derivative is a financial security with a value that is reliant upon or derived from, an underlying asset or group of assets. The use case for AD will be specically applied to calculating greeks in Option contracts or valuating Interest Rate Swaps where the interest rate yield curve can be complex. 

In the computation of derivatives, two aspects have to be taken into account; precision and speed. AD is an answer to both concerns. The goal of this feature would not prove accuracy of theoretical results, but will show efficiency through practical example. 

We expect that the implementation of this new feature will require minimum changes in the current code base. The new inputs and outputs of the financial instrument should accomodate the ones already tested and will assume that the hypothetical interest rate yield curve of the derivative will mirror f() function of securities underlying values and its derivative.      

Code directory structured can be separated into a subfolder

```
ADG4/
	advar.py
	admath.py
quant_finance/
	options.py
	interest_rate_swaps.py
tests/
	test_advar.py
	test_admath.py
	test_extension.py
docs/
	howto.md
setup.py
README.md
```

