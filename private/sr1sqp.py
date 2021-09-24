from private import pygransoConstants as pC, regularizePosDefMatrix as rPDM
from private.neighborhoodCache import nC
from private.qpSteeringStrategy import qpSS
from private.qpTerminationCondition import qpTC
from pygransoStruct import general_struct
import time
import numpy as np
from dbg_print import dbg_print,dbg_print_1
from numpy.random import default_rng
import torch
import math
from scipy import optimize


class AlgSR1SQP():
    def __init__(self):
        pass
    def sr1sqp(self,f_eval_fn, penaltyfn_obj, sr1_obj, opts, printer, torch_device):
        """
        sr1sqp:
        Minimizes a penalty function.  Note that sr1sqp operates on the
        objects it takes as input arguments and sr1sqp will modify their
        states.  The result of sr1sqp's optimization process is obtained
        by querying these objects after sr1sqp has been run.
        """
        self.f_eval_fn = f_eval_fn
        self.penaltyfn_obj = penaltyfn_obj
        self.sr1_obj = sr1_obj
        self.printer = printer

        #  "Constants" for controlling fallback levels
        #  currently these are 2 and 4 respectively
        [POSTQP_FALLBACK_LEVEL, self.LAST_FALLBACK_LEVEL] = pC.pygransoConstants()
                    
        #  initialization parameters
        x                           = opts.x0
        n                           = len(x)
        
        #  convergence criteria termination parameters
        #  violation tolerances are checked and handled by penaltyfn_obj
        self.opt_tol                     = opts.opt_tol
        self.rel_tol                     = opts.rel_tol
        step_tol                    = opts.step_tol
        ngrad                       = opts.ngrad
        evaldist                    = opts.evaldist

        #  early termination parameters
        maxit                       = opts.maxit
        maxclocktime                = opts.maxclocktime
        if maxclocktime < float("inf"):       
            t_start = time.time()
        
        self.fvalquit                    = opts.fvalquit
        # halt_on_quadprog_error      = opts.halt_on_quadprog_error
        # halt_on_linesearch_bracket  = opts.halt_on_linesearch_bracket

        #  fallback parameters - allowable last resort "heuristics"
        min_fallback_level          = opts.min_fallback_level
        self.max_fallback_level          = opts.max_fallback_level
        self.max_random_attempts         = opts.max_random_attempts

        #  steering parameters
        steering_l1_model           = opts.steering_l1_model
        steering_ineq_margin        = opts.steering_ineq_margin
        steering_maxit              = opts.steering_maxit
        steering_c_viol             = opts.steering_c_viol
        steering_c_mu               = opts.steering_c_mu
        
        #  parameters for optionally regularizing of H 
        self.regularize_threshold        = opts.regularize_threshold
        self.regularize_max_eigenvalues  = opts.regularize_max_eigenvalues
        
        self.QPsolver               = opts.QPsolver

        #  logging parameters
        self.print_level                 = opts.print_level
        print_frequency             = opts.print_frequency
        halt_log_fn                 = opts.halt_log_fn
        user_halt                   = False

        #  get value of penalty function at initial point x0 
        #  mu will be fixed to one if there are no constraints.
        self.iter            = 0
        [f,g]           = self.penaltyfn_obj.getPenaltyFunctionValue()
        self.mu              = self.penaltyfn_obj.getPenaltyParameter()
        self.constrained     = self.penaltyfn_obj.hasConstraints()

        #  The following will save all the function values and gradients for
        #  the objective, constraints, violations, and penalty function
        #  evaluated at the current x and value of the penalty parameter mu.
        #  If a search direction fails, the code will "roll back" to this data
        #  (rather than recomputing it) in order to attempt one of the fallback
        #  procedures for hopefully obtaining a usable search direction.
        #  NOTE: bumpFallbackLevel() will restore the last executed snapshot. It
        #  does NOT need to be provided.
        self.penaltyfn_at_x  = self.penaltyfn_obj.snapShot()
                                            
        #  regularizes H for QP solvers but only if cond(H) > regularize_limit
        #  if isinf(regularize_limit), no work is done

        self.torch_device = torch_device

        # get_apply_H_QP_fn   = lambda : self.getApplyH()

        # [self.apply_H_QP_fn, H_QP]   = get_apply_H_QP_fn()
        # #  For applying the normal non-regularized version of H
        # [self.apply_H_fn,*_]            = self.getApplyH()  
        
        self.sr1_update_fn          = lambda s,y: self.sr1_obj.update(s,y)

        #  function which caches up to ngrad previous gradients and will return 
        #  those which are sufficently close to the current iterate x. 
        #  The gradients from the current iterate are simultaneously added to 
        #  the cache. 
        nC_obj = nC(torch_device)
        get_nbd_grads_fn        = nC_obj.neighborhoodCache(ngrad,evaldist)
        self.get_nearby_grads_fn     = lambda : getNearbyGradients( self.penaltyfn_obj, get_nbd_grads_fn)

        # [stat_vec,self.stat_val,qps_solved, _, _] = self.computeApproxStationarityVector()
    
        if not self.constrained:
            #  disable steering QP solves by increasing min_fallback_level.
            min_fallback_level  = max(  min_fallback_level, POSTQP_FALLBACK_LEVEL )
            #  make sure max_fallback_level is at min_fallback_level
            self.max_fallback_level  = max(min_fallback_level, self.max_fallback_level)
        
        if self.max_fallback_level > 0: 
            APPLY_IDENTITY      = lambda x: x
        
        if np.any(halt_log_fn!=None):
            print("TODO: halt_log_fn in sr1sqp")
            # H_QP = None
            # get_sr1_state_fn = lambda : self.sr1_obj.getState()
            # user_halt = halt_log_fn(0, x, self.penaltyfn_at_x, np.zeros((n,1)), get_sr1_state_fn, H_QP, 1, 0, 1, stat_vec, self.stat_val, 0 )
        

        if self.print_level:
            print("hard code qps_solved and stat_val")
            qps_solved = 0
            self.stat_val = 100
            self.printer.init(self.penaltyfn_at_x,self.stat_val,qps_solved)
        

        self.rel_diff = float("inf")
        # if self.converged():
        #     return self.info
        # elif user_halt:
        #     self.prepareTermination(3)
        #     return self.info

        #  set up a more convenient function handles to reduce fixed arguments
        qpSS_obj = qpSS()
        steering_fn     = lambda penaltyfn_parts,H: qpSS_obj.qpSteeringStrategy(
                                penaltyfn_parts,    H, 
                                steering_l1_model,  steering_ineq_margin, 
                                steering_maxit,     steering_c_viol, 
                                steering_c_mu,      self.QPsolver, torch_device )

                                   
        #  we'll use a while loop so we can explicitly update the counter only
        #  for successful updates.  
        
        #  loop control variables
        self.fallback_level      = min_fallback_level
        self.random_attempts     = 0
        self.iter                = 1
        evals_so_far        = self.penaltyfn_obj.getNumberOfEvaluations()

        opts.trust_region_radius = 1.
        opts.trust_region_eta = 1e-2
        opts.trust_region_r = 1e-8

        delta = opts.trust_region_radius
        eta = opts.trust_region_eta
        r = opts.trust_region_r

        while self.iter <= maxit:
            # Call standard steering strategy to produce search direction p
            # which hopefully "promotes progress towards feasibility".
            # However, if the returned p is empty, this means all QPs failed
            # hard.  As a fallback, steering will be retried with steepest
            # descent, i.e. H temporarily  set to the identity.  If this
            # fallback also fails hard, then the standard sr1 search direction
            # on penalty function is tried.  If this also fails, then steepest
            # will be tried.  Finally, if all else fails, randomly generated
            # directions are tried as a last ditch effort.
            # NOTE: min_fallback_level and max_fallback_level control how much
            #     of this fallback range is available.

            # NOTE: the penalty parameter is only lowered by two actions:
            # 1) either of the two steering strategies lower mu and produce
            # a step accepted by the line search
            # 2) a descent direction (generated via any fallback level) is not
            # initially accepted by the line search but a subsequent
            # line search attempt with a lowered penalty parameter does
            # produce an accepted step.

            penalty_parameter_changed = False
            if self.fallback_level < POSTQP_FALLBACK_LEVEL:  
                print("Skip constrained problem temperarily")
                # if self.fallback_level == 0:
                #     apply_H_steer = self.apply_H_QP_fn;  # standard steering   
                # else:
                #     apply_H_steer = APPLY_IDENTITY; # "degraded" steering 
                
                # try:
                #     [p,mu_new,*_] = steering_fn(self.penaltyfn_at_x,apply_H_steer)
                # except Exception as e:
                #     print(e)
                #     print("PyGRANSO:steeringQuadprogFailure")
                

                # penalty_parameter_changed = (mu_new != self.mu)
                # if penalty_parameter_changed: 
                #     [f,g,self.mu] = self.penaltyfn_obj.updatePenaltyParameter(mu_new)
                
            elif self.fallback_level == 2:
                dbg_print_1( " try standard SR1 ")
                # no searching direction needed
            elif self.fallback_level == 3:
                dbg_print_1( " try steep_descent ")
                p = -g;     # steepest descent

                
            else:
                # p = np.random.randn(n,1)
                # new version of randn
                dbg_print_1( " try random search ")
                rng = default_rng()
                p = rng.standard_normal(size=(n,1))
                self.random_attempts = self.random_attempts + 1
            
            ##################################################################
            # Try trust region method
            f_prev = f      # for relative termination tolerance
            self.g_prev = g      # necessary for sr1 update

            B = self.sr1_obj.getState()
            #Decompose B into lambdas vs and r
            vals,vecs=torch.symeig(B,eigenvectors=True)
            #identify non-zero eigen values
            ind=vals>1e-6
            lambdas=vals[ind]
            vs=vecs[:,ind]
            s = trust_region_subproblem(f,g,delta,lambdas,vs,1)

            [f,g,is_feasible] = self.penaltyfn_obj.evaluatePenaltyFunction(x+s)
            y = g - self.g_prev
            # actual reduction
            ared = f_prev - f
            # predicted reduction
            pred = - (self.g_prev.t()@s + .5*s.t()@B@s)
            ap_ratio = ared/pred

            # update x to accepted iterate from trust region
            if ap_ratio > eta:
                x = x + s
            # otherwise no movement

            if ap_ratio > 0.75:
                if torch.norm(s) <= 0.8*delta:
                    pass
                    # unchanged delta
                else:
                    delta = 2*delta
            elif ap_ratio >= 0.1 and ap_ratio <= 0.75:
                pass
                # unchanged delta
            else:
                delta = .5*delta

            # if (6.26) holds
            if torch.abs(s.t()@(y-B@s)) >= r * torch.norm(s) * torch.norm(y-B@s):
                # Perform full memory SR1 update
                # This computation is done before checking the termination
                # conditions because we wish to provide the most recent (L)SR1
                # data to users in case they desire to restart.   
                self.applySr1Update(s,g,self.g_prev)
            # else no update
            self.g = g
            
            # TRUST REGION SUCCEEDED 
            ##################################################################
            
            # compute relative difference of change in penalty function values
            # this will be infinity if previous value was 0 or if the value of
            # the penalty parameter was changed
            if penalty_parameter_changed or f_prev == 0:
                self.rel_diff = float("inf")
            else:
                self.rel_diff = abs(f - f_prev) / abs(f_prev)


            # Update all components of the penalty function evaluated
            # at the new accepted iterate x and snapsnot the data.
            self.penaltyfn_at_x          = self.penaltyfn_obj.snapShot()

            # # for stationarity condition
            # [ stat_vec, self.stat_val, qps_solved, n_grad_samples,_]   = self.computeApproxStationarityVector()
                
            
            ls_evals = self.penaltyfn_obj.getNumberOfEvaluations()-evals_so_far
            
            

            if np.any(halt_log_fn!=None):
                print("TODO halt_log_fn")
                # alpha = 1.
                # user_halt = halt_log_fn(self.iter, x, self.penaltyfn_at_x, p, get_sr1_state_fn, H_QP, 
                #                         ls_evals, alpha, n_grad_samples, stat_vec, self.stat_val, self.fallback_level  )
                  
            if self.print_level and (self.iter % print_frequency) == 0:
                # step size
                alpha = torch.norm(s)
                print("hard code n_grad_samples and qps_solved")
                n_grad_samples = 0
                qps_solved = 0
                self.printer.iter(   self.iter, self.penaltyfn_at_x, self.fallback_level, self.random_attempts,  
                                    ls_evals, alpha, n_grad_samples, self.stat_val, qps_solved  );     
  
            # reset fallback level counters
            self.fallback_level  = min_fallback_level
            self.random_attempts = 0
            evals_so_far    = self.penaltyfn_obj.getNumberOfEvaluations()
    
            #  check convergence/termination conditions
            if self.converged():
                return self.info
            elif user_halt:
                self.prepareTermination(3)
                return self.info
            elif maxclocktime < float("inf") and (time.time()-t_start) > maxclocktime:
                self.prepareTermination(5)
                return self.info
            
            
            # #  if cond(H) > regularize_limit, make a regularized version of H
            # #  for QP solvers to use on next iteration
            # if self.iter < maxit:     # don't bother if maxit has been reached
            #     [self.apply_H_QP_fn, H_QP] = get_apply_H_QP_fn()
            

            self.iter = self.iter + 1   # only increment counter for successful updates
        # end while loop

        self.prepareTermination(4)  # max iterations reached

        return self.info

    #  PRIVATE NESTED FUNCTIONS
    
    

    # def computeApproxStationarityVector(self):
            
    #     #  first check the smooth case (gradient of the penalty function).
    #     #  If its norm is small, that indicates that we are at a smooth 
    #     #  stationary point and we can return this measure and terminate
    #     stat_vec        = self.penaltyfn_at_x.p_grad
    #     stat_value      = torch.norm(stat_vec)

    #     self.opt_tol = torch.as_tensor(self.opt_tol,device = self.torch_device, dtype=torch.double)
    #     if stat_value <= self.opt_tol:
    #         n_qps       = 0
    #         n_samples   = 1
    #         dist_evals  = 0
    #         self.penaltyfn_obj.addStationarityMeasure(stat_value)
    #         return [  stat_vec, stat_value, n_qps, n_samples, dist_evals ]
      
    #     #  otherwise, we must do a nonsmooth stationary point test
            
    #     #  add new gradients at current iterate to cache and then get
    #     #  all nearby gradients samples from history that are
    #     #  sufficiently close to current iterate (within a ball of
    #     #  radius evaldist centered at the current iterate x)
    #     [grad_samples,dist_evals] = self.get_nearby_grads_fn()
        
    #     #  number of previous iterates that are considered sufficiently
    #     #  close, including the current iterate
    #     n_samples = len(grad_samples)
        
    #     #  nonsmooth optimality measure
    #     qPTC_obj = qpTC()
    #     [stat_vec,n_qps,ME] = qPTC_obj.qpTerminationCondition( self.penaltyfn_at_x, grad_samples, self.apply_H_QP_fn, self.QPsolver, self.torch_device)
    #     stat_value = torch.norm(stat_vec).item()
    #     self.penaltyfn_obj.addStationarityMeasure(stat_value)
        
    #     if self.print_level > 2 and  len(ME) > 0:
    #         self.printer.qpError(self.iter,ME,'TERMINATION')
        
    #     return [ stat_vec, stat_value, n_qps, n_samples, dist_evals ]

    def converged(self):
        tf = True
        #  only converged if point is feasible to tolerance
        if self.penaltyfn_at_x.feasible_to_tol:
            if torch.norm(self.g) > self.opt_tol:
                self.prepareTermination(0)
                return tf
            elif self.rel_diff <= self.rel_tol:
                self.prepareTermination(1)   
                return tf
            elif self.penaltyfn_at_x.f <= self.fvalquit:
                self.prepareTermination(2)
                return tf
        tf = False
        return tf

    def prepareTermination(self,code):
        self.info = general_struct()
        setattr(self.info, "termination_code", code)
        if code == 8 and self.constrained:
            self.info.mu_lowest      = self.mu_lowest

    def applySr1Update(self,s,g,gprev):
                    
        y               = g - gprev
        
        update_code     = self.sr1_update_fn(s,y)
        
        if update_code > 0 and self.print_level > 1:
            print("Please check self.printer.bfgsInfo(self.iter,update_code) in sr1 sqp")
            self.printer.bfgsInfo(self.iter,update_code)
        

    # def getApplyH(self):
    #     applyH  = self.sr1_obj.applyH
    #     H       = None
    #     return [applyH, H]

    # def getApplyHRegularized(self):
    #     #  This should only be called when running full memory SR1 as
    #     #  getState() only returns the Hessian as a dense matrix in
    #     #  this case.  For L-SR1, getState() returns a struct of data.
    #     [Hr,code] = rPDM.regularizePosDefMatrix( self.sr1_obj.getState(),self.regularize_threshold, self.regularize_max_eigenvalues )
    #     if code == 2 and self.print_level > 2:
    #         self.printer.regularizeError(iter)
            
    #     applyHr  = lambda x: Hr@x
        
    #     #  We only return Hr so that it may be passed to the halt_log_fn,
    #     #  since (advanced) users may wish to look at it.  However, if
    #     #  regularization was actually not applied, i.e. H = Hr, then we can
    #     #  set Hr = [].  Users can already get H since @sr1_obj.getState
    #     #  is passed into halt_log_fn and the [] value will indicate to the 
    #     #  user that regularization was not applied (which can be checked
    #     #  more efficiently and quickly than comparing two matrices).   
    #     if code == 1:
    #         Hr = None   
        
    #     return [applyHr, Hr]


def getNearbyGradients(penaltyfn_obj,grad_nbd_fn):
    [f_grad, ci_grad, ce_grad] = penaltyfn_obj.getGradients()
    grads = general_struct()
    setattr(grads,"F",f_grad)
    setattr(grads,"CI",ci_grad)
    setattr(grads,"CE",ce_grad)
    [*_,grads_ret,dist_evals] = grad_nbd_fn(penaltyfn_obj.getX(), grads)
    return [grads_ret,dist_evals]


#Trust region subproblem solving function(Algo 3)

#Blackboxes: 
#isinrange(g is in range of B)
#findrootin(newton's method to find roots of phi(sigma)=0)
#calc_lim(calculate lim phi(sigma))
#find_p_given_sigma(find p^* given sigma^*)

def trust_region_subproblem(f,g,d,lambdas,vs,r):


    #Calculate p_u
    pu=-torch.sum((vs*(vs.t()@g))/lambdas,dim=1)
    if lambdas[0]>1e-5 and isinrange(g,vs) and torch.norm(pu)<=d:
        return pu
    if r==1:
        Q=vs
    else:
        t=2
        while lambdas[t-1]-lambdas[0]<1e-3:
            t=t+1
            if t>r:
                break
        #need to go back 1 step to find largest t that satisfies the condition
        t=t-1
        Q=vs[:,0:t]
    if(torch.sum(torch.abs(Q.t()@g)>1e-5)>0):
        sigma=findrootin(max(-lambdas[0],0),g,lambdas,vs,d,r)
        p=find_p_given_sigma(sigma,g,lambdas,vs)
        return p
    else:
        if lambdas[0]>1e-5:
            sigma=findrootin(0,g,lambdas,vs,d,r)
            p=find_p_given_sigma(sigma,g,lambdas,vs)
            return p
        else:
            lim_phi=calc_lim(-lambdas[0],g,lambdas,vs,r,d)
            print("Lim(phi)=",lim_phi)
            if lim_phi>1e-5:
                #calcluate second and third term in the formula of tau(13)
                #get lambdas and vs from t+1 to r
                lambdas_tp1_to_r=lambdas[t:r]
                vs_tp1_to_r=vs[:,t:r]
                second_term_13=torch.sum(((vs_tp1_to_r.t()@g)*(1/(lambdas_tp1_to_r-lambdas[0])))**2)
                third_term_13=((1/lambdas[0])**2)*(torch.norm(g)**2-torch.sum((vs.t()@g)**2))
                tau=torch.sqrt(d**2-second_term_13+third_term_13)
                #To deal with the plus/minus issue, need to compare the two values of tau
                #calculate first and second term in the formula of p_h(12)
                first_term_12=torch.sum((vs_tp1_to_r*(vs_tp1_to_r.t()@g))*(1/(lambdas_tp1_to_r-lambdas[0])),dim=1)
                second_term_12=(1/lambdas[0])*(g-torch.sum(vs*(vs.t()@g),dim=1))
                #p_h value for plus tau and minus tau
                ph_plus=-first_term_12+second_term_12+tau*vs[:,0]
                ph_minus=-first_term_12+second_term_12-tau*vs[:,0]
                #Calculate objective value for two values of p_h
                obj_plus=0.5*torch.sum(((vs.t()@ph_plus)**2)*lambdas) + torch.sum(ph_plus*g)
                obj_minus=0.5*torch.sum(((vs.t()@ph_minus)**2)*lambdas) + torch.sum(ph_minus*g)
                if obj_plus<obj_minus:
                    return ph_plus
                else:
                    return ph_minus
            elif lim_phi<-1e-5:
                sigma=findrootin(-lambdas[0],g,lambdas,vs,d,r)
                p=find_p_given_sigma(sigma,g,lambdas,vs)
                return p
            else:
                #get lambdas and vs from t+1 to r
                lambdas_tp1_to_r=lambdas[t:r]
                vs_tp1_to_r=vs[:,t:r]
                #calculate first and second term in the formula of p_h(12)
                first_term_12=torch.sum((vs_tp1_to_r*(vs_tp1_to_r.t()@g))*(1/(lambdas_tp1_to_r-lambdas[0])),dim=1)
                second_term_12=(1/lambdas[0])*(g-torch.sum(vs*(vs.t()@g),dim=1))
                ph=-first_term_12+second_term_12
                return ph

#isinrange(g is in range of V)
def isinrange(g,vs):
    if(torch.norm(g-vs@(vs.t()@g))<1e-3):
        return True
    else:
        return False

def findrootin(a,g,lambdas,vs,d,r):
    #define newton objective as one argument function
    def newton_obj_single(sigma):
        return newton_obj(sigma,g,lambdas,vs,r,d)
    #brentq method requires b_large s.t. f(b_large)>0
    #Closed form for large b
    b_large_1=(math.sqrt(r+1)/d)*(torch.norm(g)**2-torch.norm(vs.t()@g)**2)
    b_large=b_large_1
    for i in range(r):
       b_large_2=(math.sqrt(r+1)/d)*torch.abs(vs[:,i].t()@g)-lambdas[i]
       if b_large<b_large_2:
           b_large=b_large_2
    b_large=b_large+1
    x0,r=optimize.brentq(newton_obj_single,a,b_large,maxiter=10000,full_output=True,disp=False)
    return x0

#find_p_given_sigma(using pseudo inverse method)
def find_p_given_sigma(sigma,g,lambdas,vs):
    lambda_new=lambdas+sigma
    lambda_new_inv=lambda_new
    lambda_active=torch.abs(lambda_new)>1e-5
    lambda_new_inv[lambda_active]=1/lambda_new_inv[lambda_active]
    lambda_new_inv[torch.logical_not(lambda_active)]=0
    p=-torch.sum((vs*(vs.t()@g))*lambda_new_inv,dim=1)
    if abs(sigma)>1e-5:
        p=p-(g-vs@(vs.t()@g))/sigma
    return p

#calc_lim(calculate lim_{lambda^+} phi(sigma))
def calc_lim(sigma,g,lambdas,vs,r,d):
    #need to calculate limit at sigma->lambda^+
    sigma=sigma+1e-5
    #first calculate two terms in ||P(sigma)||^2 (11)
    first_term_11=torch.sum(((vs.t()@g)/(lambdas+sigma))**2)
    #Avoid numerical error for the first rank
    if r==1:
        second_term_11=0
    else:
        second_term_11=(1/sigma**2)*(torch.norm(g)**2-torch.norm(vs.t()@g)**2)
    #calculate ||P(sigma)||^2
    P_sigma_2=first_term_11+second_term_11
    #calculate the limit
    return 1/torch.sqrt(P_sigma_2) - 1/d

#findrootin(newton's method to find roots of phi(sigma)=0) using scipy
#First define a function to get objective value phi(sigma)
def newton_obj(sigma,g,lambdas,vs,r,d):
    #if sigma very close to 0, objective value is -1/d
    if abs(sigma)<1e-5:
        return -1/d
    #if sigma is inf, objective value is also inf
    if math.isinf(sigma):
        return math.inf
    #if sigma is close to -lambda1, return -1/d
    if abs(sigma-(-lambdas[0]))<1e-5:
        return -1/d
    
    #first calculate two terms in ||P(sigma)||^2 (20)
    first_term_11=torch.sum(((vs.t()@g)/(lambdas+sigma))**2)
    #Avoid numerical error for the first rank
    if r==1:
        second_term_11=0
    else:
        second_term_11=(1/sigma**2)*(torch.norm(g)**2-torch.norm(vs.t()@g)**2)
    #calculate ||P(sigma)||^2
    P_sigma_2=first_term_11+second_term_11
    #calculate phi(sigma)
    phi_sigma=1/torch.sqrt(P_sigma_2) - 1/d
    return phi_sigma