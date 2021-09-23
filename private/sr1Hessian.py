from pygransoStruct import general_struct
import torch

class B_obj_struct:
    
    def __init__(self,B,scaleH0):
        self.requests        = 0
        self.updates         = 0
        self.damped_requests = 0
        self.damped_updates  = 0
        self.scale_fails     = 0
        self.sty_fails       = 0
        self.infnan_fails    = 0
        self.B = B
        # self.scaleH0 = scaleH0

    # @profile
    def update(self,s,y):
        self.requests += 1
        skipped  = 0
        
        Bs = self.B@s

        # Reference: Nocedal, Jorge, and Stephen Wright. Numerical optimization. Springer Science & Business Media, 2006. 
        #             P145 SR1 method

        # there is no symmetric rank-one updating formula satisfying the secant equation
        if torch.all(y != Bs) and (y-Bs).t()@s == 0:
            skipped = 2
            self.sty_fails += 1
            print("change name for sty_fails")
            return skipped
        
        # simple update: unchanged H
        elif torch.all(y == Bs):
            return skipped

        # else:
        y_Bs = y - self.B@s
        B_new = self.B + y_Bs@y_Bs.t()/(y_Bs.t()@s)

        #  only update H if H_new doesn't contain any infs or nans
        B_vec = torch.reshape(B_new, (torch.numel(B_new),1))
        notInf_flag = torch.all(torch.isinf(B_vec) == False)
        notNan_flag = torch.all(torch.isnan(B_vec) == False)
        if notInf_flag and notNan_flag:
            self.B = B_new
            self.updates += 1
        else:
            skipped = 3
            self.infnan_fails += 1
        
        return skipped


    def getState(self):
        B_out = self.B
        return B_out

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


def sr1Hessian(B,scaleH0):
#    sr1Hessian:
#        An object that maintains and updates a sr1 approximation to the 
#        Hessian.

    B_obj = B_obj_struct(B,scaleH0)

    return B_obj