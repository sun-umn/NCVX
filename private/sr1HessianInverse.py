# import numpy as np
# from numpy.core.numeric import Inf
from pygransoStruct import general_struct
# from numpy import conjugate as conj
# from dbg_print import dbg_print_1
import torch

class H_obj_struct:
    
    def __init__(self,H,scaleH0):
        self.requests        = 0
        self.updates         = 0
        self.damped_requests = 0
        self.damped_updates  = 0
        self.scale_fails     = 0
        self.sty_fails       = 0
        self.infnan_fails    = 0
        self.H = H
        self.scaleH0 = scaleH0

    # @profile
    def update(self,s,y):
        self.requests += 1
        skipped  = 0
        
        Hy = self.H @ y

        # Reference: Nocedal, Jorge, and Stephen Wright. Numerical optimization. Springer Science & Business Media, 2006. 
        #             P145 SR1 method

        # there is no symmetric rank-one updating formula satisfying the secant equation
        if s != Hy and (s-Hy).t()@y == 0:
            skipped = 2
            self.sty_fails += 1
            print("change name for sty_fails")
            return skipped
        
        # simple update: unchanged H
        elif s == Hy:
            return skipped

        # else:
        s_Hy = s - self.H@y 
        H_new = self.H + s_Hy @ s_Hy.t()/( s_Hy.t()@y )

        #  only update H if H_new doesn't contain any infs or nans
        H_vec = torch.reshape(H_new, (torch.numel(H_new),1))
        notInf_flag = torch.all(torch.isinf(H_vec) == False)
        notNan_flag = torch.all(torch.isnan(H_vec) == False)
        if notInf_flag and notNan_flag:
            self.H = H_new
            self.updates += 1
        else:
            skipped = 3
            self.infnan_fails += 1
        
        return skipped

    def applyH(self,q):
        r = self.H @q 
        return r

    def getState(self):
        H_out = self.H
        return H_out

    def getCounts(self):
        counts = general_struct()
        setattr(counts,"requests",self.requests)
        setattr(counts,"updates",self.updates)
        setattr(counts,"damped_requests",self.damped_requests)
        setattr(counts,"damped_updates",self.damped_updates)
        setattr(counts,"scaling_skips",self.scale_fails)
        setattr(counts,"sty_fails",self.sty_fails)
        setattr(counts,"infnan_fails",self.infnan_fails)
        return counts


def bfgsHessianInverse(H,scaleH0):
#    bfgsHessianInverse:
#        An object that maintains and updates a BFGS approximation to the 
#        inverse Hessian.

    H_obj = H_obj_struct(H,scaleH0)

    return H_obj