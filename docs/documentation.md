##  Table of Contents  
[Introduction and Background](#introduction)

[How to Use](#how_to_use)

[Demo](#demo)   

[Software Organization](#org)

[Implementation](#implementation)

[Add-on Feature](#addon)

[Impact and Inclusivity](#impact)

[Future Work](#future)


<a name="introduction"/>

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
that can facilitate quick and easy numerical evaluation of derivatives through computing. In our toolbox, vector
calculation is supported. The tool box is capable of returning the Jacobian of a function with arbitrary dimension,
and is also capable of returning the partial derivative for certain variables. We integrate both forward mode and reverse mode, 
and the users are welcomed to choose the mode best suited to their scientific problem. Generally speaking,
we expect that forward mode is more suitable with lower dimensional input with higher dimensional output, and
the contrary is true for reverse mode. 


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

The Chain Rule: the derivative of a convoluted function is the product of each simple function evaluated at the value of its child function. Namely, it can be expressed in terms of the following equation. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{du}{dx}=\frac{du}{dv}\frac{dv}{dx}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{du}{dx}=\frac{du}{dv}\frac{dv}{dx}" title="\frac{du}{dx}=\frac{du}{dv}\frac{dv}{dx}" /></a>
	
The basic idea of forward mode relys on the chain rule. The final derivative can be calculated by using chain rule recursively.

Jacobian: the gradient of each element of a function's output with respect to each and every input. In other words, it's the gradient of the function at the space spanned by the inputs. The Jacobian for a function <a href="https://www.codecogs.com/eqnedit.php?latex=f:\mathbb{R}^m\to\mathbb{R}^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f:\mathbb{R}^m\to\mathbb{R}^n" title="f:\mathbb{R}^m\to\mathbb{R}^n" /></a> is

<a href="https://www.codecogs.com/eqnedit.php?latex=J=\begin{bmatrix}&space;\frac{\partial{f_1}}{\partial&space;x_1}&space;&&space;\frac{\partial{f_1}}{\partial&space;x_2}&space;&&space;...&\frac{\partial{f_1}}{\partial&space;x_m}\\&space;\frac{\partial{f_2}}{\partial&space;x_1}&space;&&space;\frac{\partial{f_2}}{\partial&space;x_2}&space;&&space;...&\frac{\partial{f_2}}{\partial&space;x_m}\\&space;...&...&...&...\\&space;\frac{\partial{f_n}}{\partial&space;x_1}&space;&&space;\frac{\partial{f_n}}{\partial&space;x_2}&space;&&space;...&\frac{\partial{f_n}}{\partial&space;x_m}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J=\begin{bmatrix}&space;\frac{\partial{f_1}}{\partial&space;x_1}&space;&&space;\frac{\partial{f_1}}{\partial&space;x_2}&space;&&space;...&\frac{\partial{f_1}}{\partial&space;x_m}\\&space;\frac{\partial{f_2}}{\partial&space;x_1}&space;&&space;\frac{\partial{f_2}}{\partial&space;x_2}&space;&&space;...&\frac{\partial{f_2}}{\partial&space;x_m}\\&space;...&...&...&...\\&space;\frac{\partial{f_n}}{\partial&space;x_1}&space;&&space;\frac{\partial{f_n}}{\partial&space;x_2}&space;&&space;...&\frac{\partial{f_n}}{\partial&space;x_m}&space;\end{bmatrix}" title="J=\begin{bmatrix} \frac{\partial{f_1}}{\partial x_1} & \frac{\partial{f_1}}{\partial x_2} & ...&\frac{\partial{f_1}}{\partial x_m}\\ \frac{\partial{f_2}}{\partial x_1} & \frac{\partial{f_2}}{\partial x_2} & ...&\frac{\partial{f_2}}{\partial x_m}\\ ...&...&...&...\\ \frac{\partial{f_n}}{\partial x_1} & \frac{\partial{f_n}}{\partial x_2} & ...&\frac{\partial{f_n}}{\partial x_m} \end{bmatrix}" /></a>

Dual numbers: a two dimentional space where an outer product is defined between any vectors <a href="https://www.codecogs.com/eqnedit.php?latex=x=(a,b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x=(a,b)" title="x=(a,b)" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=y=(c,d)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=(c,d)" title="y=(c,d)" /></a> as <a href="https://www.codecogs.com/eqnedit.php?latex=x&space;\cdot&space;y=(a&space;\cdot&space;c,a&space;\cdot&space;d&plus;c&space;\cdot&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;\cdot&space;y=(a&space;\cdot&space;c,a&space;\cdot&space;d&plus;c&space;\cdot&space;b)" title="x \cdot y=(a \cdot c,a \cdot d+c \cdot b)" /></a>. Dual number is useful when we want to include a derivative calculation, i.e. if we put the first item of a dual number as a value of a funtion, and the second item of a dual number as the derivative of a function, then the calculations  of two dual numbers <a href="https://www.codecogs.com/eqnedit.php?latex=p=(x_1,\dot{x_1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p=(x_1,\dot{x_1})" title="p=(x_1,\dot{x_1})" /></a>, and <a href="https://www.codecogs.com/eqnedit.php?latex=q=(x_2,\dot{x_2})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q=(x_2,\dot{x_2})" title="q=(x_2,\dot{x_2})" /></a> would give us the value of the calculation between the values at the first item, and the derivative of the calculation at the second item, because it turns out the calculation of the second item follows the chain rule. For example <a href="https://www.codecogs.com/eqnedit.php?latex=p&plus;q&space;=(x_1&plus;x_2,\dot{x_1}&plus;\dot{x_2})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p&plus;q&space;=(x_1&plus;x_2,x_1'&plus;x_2')" title="p+q =(x_1+x_2,x_1'+x_2')" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=pq&space;=(x_1x_2,(x_1x_2)')" target="_blank"><img src="https://latex.codecogs.com/gif.latex?pq&space;=(x_1x_2,(x_1x_2)')" title="pq =(x_1x_2,(x_1x_2)')" /></a>. When implementing forward mode, we are actually defining a basic variable as a dual number. More details can be found in the `Implementation Details` section, where our `var` is the first item and `der` is the second item of the dual number.

<a name="how_to_use"/>

# How to use

Here are the steps to use our package. First, below are the steps to download and install `ADG4`:
1. Create and activate a virtual environment, either with conda 
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

<a name="demo"/>

### Some Demos

- Some simple examples of the user interface for how to use the package is below. Running our packages involves:
- Functional inputs: A class should be called to instantiate the object. The constructor requires the following inputs: a `val` input for scaler inputs, with optional seeding for the derivative `der`. For vector inputs, `gen_vars` function should be used. If the output is a vector, then `AutoDiffVector.vconvert` should be used to vectorize the output.
- Jacobian of partial derivatives: Lastly, the function returns to the Jacobian matrix, or partial derivatives. 

Example of Creating an AutoDiffVector:
```
##import the forward mode module
import ADG4.ad as ad

a = 2.0 # Value to evaluate at
x = ad.AutoDiffVector(a) #create a AutoDiff variable with value of a

```

Simple Operation Example: Creating Functions from AD Variables
```
alpha=2.0
beta=3.0
f=alpha*x+beta # behind the scenes this calls the __add__ function and the __mul__ function respectively

# Now you can access the values and derivatives(jacobian) from the AD objects
print(f.val,f.der)

f=alpha/x-beta # behind the scenes this calls the __truediv__ function and the __sub__ function respectively
print(f.val,f.der) # print the value and derivative(jacobian)

```
Some elementary functions:

```
f=x**x #calculate pow
##print value and jacobian
print(f.val,f.der) 
```
Trig Function Examples:
```
##sin function
f = ad.sin_ad(x)
print(f.val, f.der)
##print value and jacobian
##cos function
f = ad.cos_ad(x)
##print value and jacobian
print(f.val, f.der)
##tan function
f = ad.tan_ad(x)
##print value and jacobian
print(f.val, f.der)
```
Exponential Function Example:
```
f = ad.exp_ad(x)
##print value and jacobian
print(f.val, f.der)
```

Vector implementation:
```
##multiple variables: using gen_vars method to define multiple variables together
[x,y,z,t]=ad.gen_vars([3.,np.pi,5.,3.4])
##multiple outputs: using AutoDiffVector.vconvert method to vectorize outputs
f  = ad.AutoDiffVector.vconvert([(x + y**z)/t, ad.sin_ad(x+ad.cos_ad(100*y**3)-z**t)])
##print value and jacobian
print(f.val,f.der)
##print partial derivative with respect to x
print(f.partial(x))
```
### A list of supported calculations for forward module `ad.py`

Add: AutoDiffVector.\__add__(self,other), AutoDiffVector.\__radd__(self,other)

Subtraction: AutoDiffVector.\__sub__(self,other), AutoDiffVector. \__rsub__(self,other)

Mupltiplication: AutoDiffVector.\__mul__(self,other), AutoDiffVector.\__rmul__(self,other)

Divide: AutoDiffVector.\__rturediv__(self,other), AutoDiffVector.\__rtruediv__(self,other)

Power: AutoDiffVector.\__pow__(self,other), AutoDiffVector.\__pow__(self,other)

Negative: AutoDiffVector.\__neg__(self,other)

Partial derivative with respect to a variable `vari`: AutoDiffVector.partial(self, vari)

Compare: AutoDiffVector. \__eq__(self,other), AutoDiffVector. \__ne__(self,other)

Trig functions: sin_ad(x), cos_ad(x), tan_ad(x)

Inverse trag functions: arcsin_ad(x), arccos_ad(x), arctan_ad(x)

Exponential: expa_ad(a,x), exp_ad(x)

logrithmic: loga_ad(a,x), log_ad(x)

Hyperbolic: sinh_ad(x), cosh_ad(x), tanh_ad(x)

Logistic: logistic_ad(x)



<a name="org"/>

# Software Organization

This section addresses how we organize our software package.

What will the directory structure look like?

* our directory follows a structure like this (not explicitly included in this list are our configuration files, e.g. .coverage, .gitignore: 
```
ADG4_package/
	ADG4/
		ad.py
		reverse.py
	tests/
		test_ad.py
		test_reverse.py
	setup.py
docs/
	milestone1.ipynb
	milestone2.md
	milestone2_progress.md
	documentation.md
	documentation-es.md
setup.py
README.md
README-es.md
.travis.yml
README.md
requirements.txt
```
* The directory is structured based on functionality. We have the main codes in `ADG4` with forward module `ad.py` which is the main body of our work, and an add-on package `reverse.py`. We also have a separate `\test` directory. Besides the main backage codes `ADG4_package/`, we also have `docs/` directory and some supporting files in the top directory.
 


Modules:
* We havie two modules: `ad` for implementing our forward AD functionality and `reverse` which will implement a reverse mode.
* Our current version also relies on a couple of third-party libraries to help us support specific features of the project, such as `numpy`, `copy`,`random`, and `sys`. For example, we use numpy because it is a mathematical computation library that makes it easy to build interactions between scalars, vectors, and matrices.  We have include these libraries in `requirements.txt` so users can install them easily with `pip install -r requirements.txt` on their machine orvirtual environment.

Test Suite:
* We are using both TravisCI and CodeCov as part of our test suite.
* The project will leverage the `pytest` module to test, and will live in a separate directory structure as seen above. The test suite will be run automatically via TravisCI everytime we push a change into our branch. We set the test suite up in the `.travis.yml` file. Each time code is pushed, they both will run all the tests in the `tests/` dir.
* The project repo has a badge reporting on the coverage of our code from Codecov, so we can easily tell how many tests are passing.

Package Distribution:
* Our package is pip installable using the editable option pointing to a local file system. It will be pip installable when it is finally published.
* Specific step by step instructions for how to download and install our package are provided above in the How to Use section.
* As a brief summary, `ADG4` can be downloaded and installed by creating and activating a virtual environment, downloading our repository (`git clone git@github.com:CS107-gharvar4d/cs107-FinalProject.git`), and navigating into the repo folder with `cd cs107-FinalProject`. 
* Then, install the requirements with `pip install -r requirements.txt`, and install the `ADG4` package with `pip install --editable ./code` (code is the name of the directory where it lives).
* Now you can use the command `import ADG4.ad as ad` and you are ready to use our package!

Sofware Packaging:
* We use SetupTools (setup.py) to package our software. That way it can handle downloading dependencies and setup processes.

Other Considerations:
* As noted in the project instructions we will also include a broader impact statement for our library in README.md. This will consider the accessibility of our software library to different groups of people and ensure that it is accessible and usable to a wide and representative population. We also include a prototype of the documentation in Spanish to help the package to gain further inclusivity. The credit goes to Rodrigo Vargas in our group.

<a name="implementation"/>

# Implementation Details

The core data structure is an AutoDiffVector which maintains two core pieces of data:
the current value, and the the derivatives with regard to *all* the independent variables.
Eg the partial derivatives with respect to each input variable.


## Core Classes 
### AutoDiffVector
```
AutoDiffVector
  fields:
    - val: the currently computed value, it is either a scaler or a vector.
    - der: the jacobian with respect to all the input (aka. independent) variables. It is either a scaler, vector, or a matrix.
  methods:
    - partial(vari)
    - __mul__, __add__, .. call the associated ADFunctions to return new AutoDiffVector
```

### Managing derivatives
At any point, the method `partial(vari=v)` can be called on an AutoDiffVector get the derivative with respect to any input variables.


## ADFunctions
ADFunctions accept one AutoDiffVector and output a new AutoDiffVector with an updated `val` and `der`.
```
ADFunction
  input: One AutoDiffVectors
  output: A new AutoDiffVector with appropriate val, der
```

### ADFunction Implementations
Since ADFunctions do not contain internal state, and cannot be loaded as dunder methods (e.g., sin, cos), for now it seems to make the most sense to have them as direct functions.
Elementary functions like sin, cos, etc will each have a corresponding `ADFunction` implementation that stores gradients
and maintains logic depending on whether the input is a vector, matrix etc. For functions such as multiplication and addition,
which have implicit functionality in Python, we will overwrite the underlying dunder method so that when called upon an `AutoDiffVector`,
these functions will evaluate and update the derivatives as described above.

No external dependencies are required at this time, except `numpy` package.

Although ADFunctions are considered as a class, when implementing it, we choose not to bind these functions together as a class. We just define them separately to guarantee easy access to them.


### Scalar and Vector
The `AutoDiffVector` is resemble to a single output function with multiple inputs. The multiple inputs should be defined together as shown in the example below.
`ADFunction`s handles `AutoDiffVector`s of any type. Namely, multiple inputs is supported. Considering that different elements of vector outputs is naturally different, we did not make throughly test `ADFunction`s to guarantee them support multiple outputs, although multiple output functions should be supported based on our implementation.

The output can be vectorized using the class method vconvert after calculating each line of output separately, as shown in the example below.


####  Scalars, Vectors example:

```
v = AutoDiffVector(1) # a scalar input with scaler output
z = ad.sin_ad(v)

##multiple variables: using gen_vars method to define multiple variables together
[x,y,z,t]=ad.gen_vars([3.,np.pi,5.,3.4])

##multiple outputs: using AutoDiffVector.vconvert method to vectorize outputs
f  = ad.AutoDiffVector.vconvert([(x + y**z)/t, ad.sin_ad(x+ad.cos_ad(100*y**3)-z**t)])

##multiple output functions f is still a AutoDiffVector, so we can still apply ADFunctions to it
print(ad.sin_ad(f))
print(f+f)

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

<a name="addon"/>

# Add-on Feature

## Reverse Mode Differentiation. 

### A brief background introduction to reverse mode
We've been focusing in forward mode, where we carry derivatives along and traverse the graph at each node. But there is another method in which we build a graph and store a partial derivative at each node and contrary to forward mode, we leverage a backpropagation technique. We generate the forward trace with the input of the equations, however the derivative is not accumulated at this point, we just calculate the partial derivative and store them. After we completed the full eqaution, namely we have stored the whole graph of the trace of the calculation with the partial derivatives along the way, the derivatives with respect to the original variables can be calculated backforward along the graph, and the final result will be retrieved when we comes to the origin of the equation (the independent variables).

Reverse mode utilizes similar element formulas to the ones implemented in forward mode. The chain rule still shows is power in its implementation, but dual number is not used. As an mathematical example, consider

<a href="https://www.codecogs.com/eqnedit.php?latex=f=sin(x_1^2)&plus;x_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f=sin(x_1^2)&plus;x_2" title="f=sin(x_1^2)+x_2" /></a> 

When moving forward with the equation, at step 1 we calculate and store <a href="https://www.codecogs.com/eqnedit.php?latex=x_1'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_1'" title="x_1'" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=x_2'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_2'" title="x_2'" /></a>. At step 2, we store <a href="https://www.codecogs.com/eqnedit.php?latex=x_1^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_1^2" title="x_1^2" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=v_1=x_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_1=x_1" title="v_1=x_1" /></a>. At step 3, we store <a href="https://www.codecogs.com/eqnedit.php?latex=sin(v_2)'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?sin(v_2)'" title="sin(v_2)'" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=v_2=v_1^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_2=v_1^2" title="v_2=v_1^2" /></a>. When calculating the partial derivative with respect to <a href="https://www.codecogs.com/eqnedit.php?latex=v_1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?v_1" title="v_1" /></a>, we move backward and multiply together the derivatives and get <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d(sin(v_2))}{dv_2}\frac{dv_1^2}{dv_1}\frac{dx^2}{dx}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d(sin(v_2))}{dv_2}\frac{dv_1^2}{dv_1}\frac{dx^2}{dx}" title="\frac{d(sin(v_2))}{dv_2}\frac{dv_1^2}{dv_1}\frac{dx^2}{dx}" /></a>. For the partial derivative with repect to <a href="https://www.codecogs.com/eqnedit.php?latex=x_2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_2" title="x_2" /></a>, we directly trace back to <a href="https://www.codecogs.com/eqnedit.php?latex=x_2'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_2'" title="x_2'" /></a> and get the derivative as 1.


### Software organization for add-on feature
As has been shown in the `Software Organization` section, we have a module `reverse.py` separately for the reverse mode, and a test file `test_reverse.py` for it separately from the forward mode.

### How to use reverse mode
It is quite similar to the feature of forward mode and self-evident.
Example of Creating an AutoDiffVector:
```
##import the reverse mode module
import ADG4.reverse as rev
#Here we give the users a choice to explicity give a name to the independent variables considering the implementation do not give an explicit order to the variables.
#Our implementation naturally support vector inputs
x = rev.AutoDiffReverse(3, name='x')
y = rev.AutoDiffReverse(4, name='y')
z = rev.AutoDiffReverse(9, name='z')

```

Simple Operation Example: Creating Functions from AD Variables
```
m = x + y
n = m * z + x
p = x * x * x

assert n.partial(x) == 10
assert n.partial(y) == 1 * z.val
assert n.partial(z) == m.val #.val is the value of the function, which is the same to the forward mode
assert m.partial(x) == 1
assert m.partial(y) == 1
assert p.partial(x) == 3 * 3 * 3


```
Some elementary functions:

Power:

```
f=x**x #calculate pow
print(f.val,f.partial(x)) 
```
Trig Function Examples:
```
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
```
Exponential Function Example:
```
f = rv.exp_rv(x)
##print value and jacobian
print(f.val, f.partial(x))
```

Basically we have all the same functions to forward mode, with slight differents in implementation. For reverse mode, we do not have compare functions considering the equation is path dependent. But users can still use the default compare functions of python, which compare the reference of two instances.

### Implementation Details

<a name="impact"/>

# ADG4 Impact and Inclusivity. 

It is important to consider the potential effects on building software and make it available to all without distinction. Our philosophy is to distribute software to anyone to any purpose and make efforts to develop it in a collaborative public matter. A good start for this is the fact that all of our source code resides in Github, an open platform where everyone can collaborate. Further, we believe that individuals across borders and languages should be able to collaborate on and use our project with equal access. We thus include documentation for non-English speakers such as Spanish, with more to come in the future.

Our group accepts two common workflows for collaboration:

1. Basic Shared Repository

Clone our repo and update with `git pull origin master`, then create a working branch with `git checkout -b MyNewBranch` and make any changes to it before staging. 
Commit locally and upload the changes (including your new branch) to GitHub with `git push origin MyNewBranch`

Then, navigate to main on GitHub where you should now see your new branch. Click on “Pull Request” button and “Send Pull Request”

2. Fork repo and pull

We can assign rights to “Collaborators”. Even though collaborators do not have push access to upstream, we accept Pull Requests (PRs) from them, reviews and then merge changes into main repo if approved.

All ADG4 collaborators should adhere to NeurIPS standards, remain vigilant and assest the impact of their code for unethical behaviour or illegal use. Some software applications where this might happen are security or privacy. If any misuse is known the project leader should be contacted immediately. We also want to ensure that our software is welcoming to individuals across areas of applied computing. With this in mind, we encourage collaborators to avoid using certain computing terms that can be considered offensive or exclusionary by some groups. A list of such terms and suggested alternatives can be found on the [Association for Computing Machinery](https://www.acm.org/diversity-inclusion/words-matter) website. Some examples include avoiding gendered pronouns and racially charged words such as blacklist/whitelist.

Finally, we are aware of [the gender gap in open source contributions](https://medium.com/intuit-engineering/open-source-where-are-the-women-ae20623529ca) and want to personally encourage female developers to collaborate on our software if they are interested in doing so. Since [data](https://www.techrepublic.com/article/diversity-why-open-source-needs-to-work-on-it-in-2020/) has indicated that access to resources would make women more likely to contribute to open-source projects, we would be happy to provide any additional materials or instructions personally if the materials contained in this repository are insufficient.

<a name="future"/>

## Future Work

While our primary add-on feature is reverse mode, there is also strong interest in the group to continue working on additional extensions for this project. Broadly speaking, these ideas apply autodifferentiation to real world problems that can be done more efficiently. Since these ideas are not fully crystallized, more work will likely need to be done to hone the implementation. A couple of example include:

1. Pricing Exotic Derivatives

There have been emerging conversations between Academia and the Quantitative Finance Industry of using this type of techniques for valuating derivatives. A derivative is a financial security with a value that is reliant upon or derived from, an underlying asset or group of assets. The use case for AD will be specifically applied to calculating Greeks in Option contracts or evaluating Interest Rate Swaps where the interest rate yield curve can be complex. In the computation of derivatives, two aspects have to be taken into account; precision and speed. AD is an answer to both concerns. The goal of this feature would not prove accuracy of theoretical results, but will show efficiency through practical examples. 

2. Parameter Fitting on a Time-Dependent System

Another potential future extension involves fitting an algorithm for a time dependent system. Given a governing equation with a few undecided parameters and a set of observational data, the AD integrates the equation over time and compares the result against the observations. This type of algorithm can be applicable in the real world for use cases such as complex regressions, imaging and revolve spectral mixtures.  
