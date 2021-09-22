import numpy as np
import numpy.linalg as LA
import math
from numpy import conjugate as conj
import torch
from dbg_print import dbg_print_1


# @profile
def linesearchWeakWolfe( x0, f0, grad0, d, f_eval_fn, obj_fn, c1 = 0, c2 = 0.5, fvalquit = -np.inf, eval_limit = np.inf, step_tol = 1e-12, init_step_size = 1, linesearch_maxit = np.inf, is_backtrack_linesearch = False, torch_device = torch.device('cpu')):
    """
    linesearchWeakWolfe:
        Line search enforcing weak Wolfe conditions, suitable for minimizing 
        both smooth and nonsmooth functions.  This routine is a slightly 
        modified version of linesch_ww.m from HANSO 2.1, to faciliate a few 
        different input and output arguments but the method itself remains 
        unchanged.  The function name has been changed so that they are not
        mistakenly used in lieu of one another.  
        NOTE: the values assigned to output argument "fail" have been changed 
                so that all error cases are assigned positive codes.
    """
    # is_backtrack_linesearch = False

    # eval_limit = 25
    # dbg_print_1("hard coding eval_limit = %d: initial t = 10"%eval_limit)
    
    # d_rescale = d.detach().clone()

    alpha = 0  # lower bound on steplength conditions
    xalpha = x0.detach().clone()
    falpha = f0
    gradalpha = grad0.detach().clone() # need to pass grad0, not grad0'*d, in case line search fails
    beta = float('inf')  # upper bound on steplength satisfying weak Wolfe conditions

    # gradbeta = torch.empty(x0.shape,device=device)
    # gradbeta[:] = float('nan')
    g0 = (torch.conj(grad0.t()) @ d).item() 
    dnorm = torch.norm(d).item()
    # t = 1  # important to try steplength one first
    # t = 1e-2  # important to try steplength one first
    t = init_step_size
    n_evals = 0
    nexpand = 0
    maxit = min(eval_limit,linesearch_maxit)

    # the following limit is rather arbitrary
    # don't use HANSO's nexpandmax, which could much larger, since BFGS-SQP 
    # will automatically reattempt the line search with a lower penalty 
    # parameter if it terminates with the "f may be unbounded below" case.
    nexpandmax = max(10, round(math.log2(1e5/dnorm)))  # allows more if ||d|| small

    test_flag = 0

    while (beta - alpha) > (torch.norm(x0 + alpha*d).item()/dnorm)*step_tol and n_evals < maxit:
        x = x0 + t*d

        if is_backtrack_linesearch:
            [f,is_feasible] = f_eval_fn(x)
            # random setting, avoid 2nd wolfe condition
            gtd = 2*(torch.conj(grad0.t()) @ d)
        else:
            [f,grad,is_feasible] = obj_fn(x)
            gtd = torch.conj(grad.t()) @ d

        if torch.is_tensor(f):
            f = f.item()
        n_evals = n_evals + 1
        if is_feasible and not np.isnan(f) and f <= fvalquit and not np.isinf(f): 
            fail = 0
            alpha = t  # normally beta is inf
            xalpha = x.detach().clone()
            [f,grad,is_feasible] = obj_fn(x)
            falpha = f
            gradalpha = grad.detach().clone()
            # return [alpha, xalpha, falpha, gradalpha, fail, beta, gradbeta, n_evals]
            return [alpha, xalpha, falpha, gradalpha, fail] 
        
        #  the first condition must be checked first. NOTE THE >=.
        if f >= f0 + c1*t*g0 or np.isnan(f): # first condition violated, gone too far
            beta = t
            # gradbeta = grad.detach().clone() # discard f
            test_flag = 1

        
        elif not is_backtrack_linesearch and gtd <= c2*g0 or torch.isnan(gtd): # second condition violated, not gone far enough
                alpha = t
                xalpha = x.detach().clone()
                falpha = f
                gradalpha = grad.detach().clone()



        else:   # quit, both conditions are satisfied
            fail = 0
            alpha = t
            xalpha = x.detach().clone()
            if is_backtrack_linesearch:
                [f,grad,is_feasible] = obj_fn(x)
            falpha = f
            gradalpha = grad.detach().clone()
            beta = t
            # gradbeta = grad.detach().clone()
            # dbg_print_1("final step size t = %f "%t)
            # return [alpha, xalpha, falpha, gradalpha, fail, beta, gradbeta, n_evals] 
            return [alpha, xalpha, falpha, gradalpha, fail] 
        
        #  setup next function evaluation
        if beta < np.inf:
            t = (alpha + beta)/2 # bisection
        elif nexpand < nexpandmax:
            nexpand = nexpand + 1
            t = 2*alpha  # still in expansion mode
        else:
            break # Reached the maximum number of expansions

    # end loop
    # Wolfe conditions not satisfied: there are two cases
    if beta == np.inf: # minimizer never bracketed
        fail = 2
    else: # point satisfying Wolfe conditions was bracketed
        # dbg_print_1("final step size t = %f "%t)
        dbg_print_1("wolfe condition %d fails"%test_flag)
        fail = 1
    

    #####################################################################
    if is_backtrack_linesearch:
        dbg_print_1("return t when line searhc fails:")
        alpha = t
        xalpha = x.detach().clone()
        [f,grad,is_feasible] = obj_fn(x)
        falpha = f
        gradalpha = grad.detach().clone()
        beta = t
        # gradbeta = grad.detach().clone()
        dbg_print_1("final step size t = %f \n"%t)

    return [alpha, xalpha, falpha, gradalpha, fail]                              