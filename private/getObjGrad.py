from dbg_print import dbg_print_1
import numpy as np
# from pygransoStruct import GeneralStruct
import torch


def getObjGrad(nvar,var_dim_map,f,X, torch_device):
    # f_grad = genral_struct()
    f.backward(retain_graph=True)
    # transform f_grad form matrix form to vector form
    # f_grad_vec = np.zeros((nvar,1))

    
    dbg_print_1('Using device in getObjGrad')
    f_grad_vec = torch.zeros((nvar,1),device=torch_device, dtype=torch.double)

    curIdx = 0
    # current variable, e.g., U
    for var in var_dim_map.keys():
        # corresponding dimension of the variable, e.g, 3 by 2
        dim = var_dim_map.get(var)
        grad_tmp = getattr(X,var).grad
        varLen = np.prod(dim)
        f_grad_reshape = torch.reshape(grad_tmp,(varLen,1))
        f_grad_vec[curIdx:curIdx+varLen] = f_grad_reshape
        curIdx += varLen

        # preventing gradient accumulating
        getattr(X,var).grad.zero_()
    return f_grad_vec
# @profile
def getObjGradDL(nvar,model,f, torch_device):
    # f_grad = genral_struct()
    f.backward()
    # transform f_grad form matrix form to vector form
    f_grad_vec = torch.zeros((nvar,1),device=torch_device, dtype=torch.double)

    curIdx = 0
    parameter_lst = list(model.parameters())
    for i in range(len(parameter_lst)):
        # print(parameter_lst[i].grad.shape)
        f_grad_reshape = torch.reshape(parameter_lst[i].grad,(-1,1))
        f_grad_vec[curIdx:curIdx+f_grad_reshape.shape[0]] = f_grad_reshape
        curIdx += f_grad_reshape.shape[0]

    return f_grad_vec