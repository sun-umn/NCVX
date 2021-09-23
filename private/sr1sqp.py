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

        [stat_vec,self.stat_val,qps_solved, _, _] = self.computeApproxStationarityVector()
    
        if not self.constrained:
            #  disable steering QP solves by increasing min_fallback_level.
            min_fallback_level  = max(  min_fallback_level, POSTQP_FALLBACK_LEVEL )
            #  make sure max_fallback_level is at min_fallback_level
            self.max_fallback_level  = max(min_fallback_level, self.max_fallback_level)
        
        if self.max_fallback_level > 0: 
            APPLY_IDENTITY      = lambda x: x
        
        if np.any(halt_log_fn!=None):
            print("TODO: halt_log_fn in sr1sqp")
            H_QP = None
            get_sr1_state_fn = lambda : self.sr1_obj.getState()
            user_halt = halt_log_fn(0, x, self.penaltyfn_at_x, np.zeros((n,1)), get_sr1_state_fn, H_QP, 1, 0, 1, stat_vec, self.stat_val, 0 )
        

        if self.print_level:
            self.printer.init(self.penaltyfn_at_x,self.stat_val,qps_solved)
        

        self.rel_diff = float("inf")
        if self.converged():
            return self.info
        elif user_halt:
            self.prepareTermination(3)
            return self.info

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
                if self.fallback_level == 0:
                    apply_H_steer = self.apply_H_QP_fn;  # standard steering   
                else:
                    apply_H_steer = APPLY_IDENTITY; # "degraded" steering 
                
                try:
                    [p,mu_new,*_] = steering_fn(self.penaltyfn_at_x,apply_H_steer)
                except Exception as e:
                    print(e)
                    print("PyGRANSO:steeringQuadprogFailure")
                

                penalty_parameter_changed = (mu_new != self.mu)
                if penalty_parameter_changed: 
                    [f,g,self.mu] = self.penaltyfn_obj.updatePenaltyParameter(mu_new)
                
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
            s = trust_region_subproblem(g,B,delta)

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

            # for stationarity condition
            [ stat_vec, self.stat_val, qps_solved, n_grad_samples,_]   = self.computeApproxStationarityVector()
                
            
            ls_evals = self.penaltyfn_obj.getNumberOfEvaluations()-evals_so_far
            
            
        
            if np.any(halt_log_fn!=None):
                print("TODO halt_log_fn")
                alpha = 1.
                user_halt = halt_log_fn(self.iter, x, self.penaltyfn_at_x, p, get_sr1_state_fn, H_QP, 
                                        ls_evals, alpha, n_grad_samples, stat_vec, self.stat_val, self.fallback_level  )
                  
            if self.print_level and (self.iter % print_frequency) == 0:
                # step size
                alpha = torch.norm(s)
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
    
    

    def computeApproxStationarityVector(self):
            
        #  first check the smooth case (gradient of the penalty function).
        #  If its norm is small, that indicates that we are at a smooth 
        #  stationary point and we can return this measure and terminate
        stat_vec        = self.penaltyfn_at_x.p_grad
        stat_value      = torch.norm(stat_vec)

        self.opt_tol = torch.as_tensor(self.opt_tol,device = self.torch_device, dtype=torch.double)
        if stat_value <= self.opt_tol:
            n_qps       = 0
            n_samples   = 1
            dist_evals  = 0
            self.penaltyfn_obj.addStationarityMeasure(stat_value)
            return [  stat_vec, stat_value, n_qps, n_samples, dist_evals ]
      
        #  otherwise, we must do a nonsmooth stationary point test
            
        #  add new gradients at current iterate to cache and then get
        #  all nearby gradients samples from history that are
        #  sufficiently close to current iterate (within a ball of
        #  radius evaldist centered at the current iterate x)
        [grad_samples,dist_evals] = self.get_nearby_grads_fn()
        
        #  number of previous iterates that are considered sufficiently
        #  close, including the current iterate
        n_samples = len(grad_samples)
        
        #  nonsmooth optimality measure
        qPTC_obj = qpTC()
        [stat_vec,n_qps,ME] = qPTC_obj.qpTerminationCondition( self.penaltyfn_at_x, grad_samples, self.apply_H_QP_fn, self.QPsolver, self.torch_device)
        stat_value = torch.norm(stat_vec).item()
        self.penaltyfn_obj.addStationarityMeasure(stat_value)
        
        if self.print_level > 2 and  len(ME) > 0:
            self.printer.qpError(self.iter,ME,'TERMINATION')
        
        return [ stat_vec, stat_value, n_qps, n_samples, dist_evals ]

    def converged(self):
        tf = True
        #  only converged if point is feasible to tolerance
        if self.penaltyfn_at_x.feasible_to_tol:
            if self.stat_val <= self.opt_tol:
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