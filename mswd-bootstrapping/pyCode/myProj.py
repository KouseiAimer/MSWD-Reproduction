import torch
import numpy as np
from pyCode.global_var import device

def phi_rho(rho, vals, lam):
    """
        compute the value of phi(rho) (\|v\|_1 <= lam)
    """
    abs_v_sorted = vals['abs_v_sorted']
    idx = torch.where(abs_v_sorted > rho)
    phi = torch.norm(abs_v_sorted[idx] - rho, 1)**2 - lam**2*torch.norm(abs_v_sorted[idx]-rho, 2)**2 # better calculation precision
    return phi

def get_values(V):
    """ output some useful quantities """
    V = V.to(device)  
    abs_v = torch.abs(V)
    abs_v_sorted, indices= abs_v.sort(descending=True, dim=0) # V[indices] = values
    s_l1 = torch.cumsum(abs_v_sorted, dim=0)
    w_l2 = torch.cumsum(abs_v_sorted**2, dim=0)
    return {'abs_v_sorted': abs_v_sorted, 's_l1': s_l1, 'w_l2': w_l2, 'indices': indices}

def get_V(V, lam):
    V = V.to(device)
    dim = V.size(0)
    ones = torch.ones(dim,1,device=device)
    abs_V = torch.abs(V)
    V1 = abs_V - lam*ones
    indices = (V1>0)
    VV = torch.zeros(dim,1,device=device)
    VV[indices] = V1[indices]
    VV = VV / torch.norm(VV, p=2)
    return VV

def myProj(V, lam, thresh=1e-6, maxIter=100):
    V = V.to(device)
    if torch.norm(V, p=1) <= (lam*torch.norm(V, p=2)):
        VV = V / torch.norm(V, p=2)
    else:
        sign_v = torch.sign(V)
        dim = V.size(0)       
        vals = get_values(V)
        abs_v_sorted = vals['abs_v_sorted']
        I1 = torch.sum(abs_v_sorted >= abs_v_sorted[0])   
        if I1 > (lam**2):
            if I1 == 1:
                print('UserWarning: there exists no solution!')
                return None
            else:              
                VV = torch.zeros(dim, 1, device=device)
                inds = vals['indices'][0:I1]
                VV[inds[1:I1]] = (lam*(I1-1) - ((I1-1)*(I1-lam**2))**0.5) / (I1*(I1-1))
                VV[inds[0]] = lam - (I1-1)*VV[inds[1]]
                VV = VV * sign_v
        elif I1 == (lam**2):
            VV = torch.zeros(dim, 1, device=device)
            VV[vals['indices'][0:I1],0] = torch.ones(I1,1,device=device)/I1**0.5
            VV = VV * sign_v
        else:
            l = 0 # phi(l)>0
            temp = torch.topk(torch.unique(abs_v_sorted), k=2).values
            r = temp[1] # phi(r) < 0
            phi_l = phi_rho(l, vals, lam)
            phi_r = phi_rho(r, vals, lam)
            if torch.abs(phi_l) <= torch.abs(phi_r):
                rho = l
                phi_rho_val = phi_l
            else:
                rho = r
                phi_rho_val = phi_r
            error = torch.abs(phi_rho_val)
            count = 0
            while (r-l)>thresh and error>thresh and (phi_l-phi_r)>thresh and count<=maxIter:
                count += 1
                rho_S = r - (l-r)/(phi_l - phi_r)*phi_r # l < rho_S < r; phi(rho_S) < 0
                Il = torch.sum(abs_v_sorted >= l)
                s = vals['s_l1'][Il-1]
                w = vals['w_l2'][Il-1]
                if (Il*w - s**2) <= 0: # to avoid numerical issues caused by calculation precision; this quantity should be nonnegative
                    rho_Q = s / Il
                else:
                    rho_Q = (s - lam*((Il*w - s**2)/(Il-lam**2))**0.5) / Il # phi(rho_Q) >= 0
                rho = (rho_S+rho_Q) / 2
                phi_rho_val = phi_rho(rho, vals, lam)
                if torch.sum(torch.logical_and(abs_v_sorted>l, abs_v_sorted<rho_Q))==0 and rho_Q < abs_v_sorted[0]:
                    rho = rho_Q # phi(rho) = 0
                    VV = get_V(V, rho)
                    VV = VV * sign_v
                    return VV
                elif phi_rho_val >= 0:
                    l = rho
                    r = rho_S
                else:
                    l = rho_Q
                    r = rho
                error = torch.abs(phi_rho_val)
                phi_l = phi_rho(l, vals, lam)
                phi_r = phi_rho(r, vals, lam)
            if rho < abs_v_sorted[0]:
                VV = get_V(V, rho)
                VV = VV * sign_v
            else:
                VV = torch.where(torch.abs(V) == abs_v_sorted[0], 1, 0)
                VV = VV / torch.norm(VV)
                VV = VV * sign_v
    return VV
