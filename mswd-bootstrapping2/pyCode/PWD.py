"""
    compute the projected Wasserstein distance 
    ISIT(2021), "Two-sample Test using Projected Wasserstein Distance: Breaking the Curse of Dimensionality"
"""

import torch
import numpy as np
import ot ## need to pip install POT=='0.8.2'; weights need to be at least nonnegative
# the marginal weights p1 and p2 can be on cpu (when dist_mat is on gpu)
# for general case, p1 and p2 should be on the same device as dist_mat
from pyCode.global_var import device

def pdist(X, Y, V):
    """ 
        compute the distance matrix of projected variables
    """
    x_proj = torch.matmul(X, V)
    y_proj = torch.matmul(Y, V)
    dist_mat = torch.cdist(x_proj, y_proj, p=2)
    return dist_mat

def update_OT(X, Y, V, p1, p2):
    """
        compute the optimal coupling (transport plan)
        linear programming
    """
    dist_mat = pdist(X, Y, V)
    T_coupling = ot.emd(p1, p2, dist_mat)
    T_coupling = T_coupling.to(device)
    return T_coupling

def Rie_gradient(V, A):
    """ compute the Riemannian gradient, project A onto the tangent space of V """
    grad = A - torch.matmul(V, torch.matmul(V.T, A)+torch.matmul(A.T, V))/2
    return grad

def update_V_PWD(X, Y, T_coupling, old_V, lam, learn_rate):
    """ update the projection V """
    n1 = X.size(0)
    n2 = Y.size(0)
    p = X.size(1)
    # compute the sub-differential
    dist_mat = pdist(X, Y, old_V)
    dist_mat = torch.sqrt(dist_mat**2 + 1e-6)
    PP = T_coupling/dist_mat
    G = 0
    for i in range(n1):
        x_temp = X[i,:].repeat(n2, 1).T #p*n2
        PP_temp = torch.diag(PP[i,:])
        G = G + torch.matmul(torch.matmul(x_temp-Y.T, PP_temp), (x_temp-Y.T).T)
    A = torch.matmul(G, old_V)
    A = A - lam * torch.sign(old_V)
    # project onto the tagent space
    grad = Rie_gradient(old_V, A)
    # retraction
    V, _ = torch.linalg.qr(old_V + learn_rate*grad)
    return V

def PWD(X, Y, V0, p1, p2, lam=0, learn_rate=100, thresh=1e-6, max_iter=100):
    """
        optimize the projected Wasserstein distance
        output the optimal coupling T_coupling and projection V 
    """
    X = X.to(device)
    Y = Y.to(device)
    V0 = V0.to(device)
    p1 = p1.to(device)
    p2 = p2.to(device)
    old_V = V0.clone()
    iter_id = 0
    error = 1
    while error > thresh and iter_id <= max_iter:
        iter_id = iter_id+1
        learn_rate = learn_rate / iter_id**0.5
        # learn_rate = learn_rate * 0.5
        T_coupling = update_OT(X, Y, old_V, p1, p2)
        V = update_V_PWD(X, Y, T_coupling, old_V, lam=lam, learn_rate=learn_rate) 
        error = torch.linalg.norm(V-old_V) / (1+torch.linalg.norm(old_V))
        old_V = V.clone()
    dist_mat = pdist(X, Y, V)
    # pwd = ot.emd2(p1, p2, dist_mat).to('cpu')
    pwd = torch.sum(ot.emd(p1, p2, dist_mat) * dist_mat).to('cpu')
    return pwd
