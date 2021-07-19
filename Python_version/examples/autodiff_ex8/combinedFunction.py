import numpy as np
from numpy import linalg as LA
from numpy.core.fromnumeric import transpose
import torch


class f_gradStruct:
    pass

class ciStruct:
    pass

class ci_gradStruct:
    class c1:
        pass
    class c2:
        pass

def combinedFunction(X):
    
    # input variable, matirx form
    # u = torch.tensor(X.u)
    u = X.u
    u.requires_grad_(True)
     
    # objective function
    # f must be a scalar. not 1 by 1 matrix
    

    f = torch.sum(u**2)**0.5

    # f_grad.u = u/np.sum(np.square(u))**0.5
        
    # inequality constraint 
    ci = None
    ci_grad = None
    
    # equality constraint 
    ce = None
    ce_grad = None

    return [f,ci,ci_grad,ce,ce_grad]