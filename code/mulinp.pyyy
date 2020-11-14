import numpy as np
import copy
from ad import *

class AutoDiff2(AutoDiffToy):
    
    def __init__(self,a,d):
        self.val=a
        self.der=d

def gen_seed(a,p):
    p=p/np.sqrt(np.sum(np.power(p,2.)))
    return [AutoDiff2(aa,dd) for aa,dd in zip(a,p)]

def jacobian(input_list,formular,input_var):
    jacob=np.zeros(len(input_var))
    for ii in range(len(input_var)):
        p=np.zeros([len(input_var)])
        p[ii]=1.
        print(",".join(input_list)+"="+"gen_seed("+str(input_var)+","+",".join(str(p).split(' '))+")")
        exec(",".join(input_list)+"="+"gen_seed("+str(input_var)+","+",".join(str(p).split(' '))+")",globals())
        exec("f="+formular,globals())
        jacob[ii]=f.der
    new=AutoDiff2(f.val,jacob)
    return new


f = jacobian(["x","y","z","t"],"x**2+sin_ad(y*exp_ad(z)+cos_ad(t))",[3,4,1,-100])
print(f.val, f.der)

f = jacobian(["x","y","z","t"],"x+y+z+t",[3,4,1,-100])
print(f.val, f.der)
