
## implement the proposed test given data 
import torch
import numpy as np
from pyCode.maxSlicedWD_L0_L1Approx import maxSlicedWDL0_L1Approx, maxSlicedWDL0_L1Approx_bootstrap
from pyCode.myProj import myProj
from pyCode.global_var import device

def mswdTest(data1, data2, lam, alpha=0.05, B=500, candidate_adaptive=True, n_l1=10, reps=10):
    # data1, data2: tensor
    # lam: l0 sparsity parameter
    # alpha: significance level; B: the number of bootstrapping samples
    # reps: the number of different initial values for optimization
    # candidate_adaptive: whether adaptively determine the sequence of l1 sparsity parameters
    # n_l1: the number of l1 parameters

    data1 = data1.to(device)
    data2 = data2.to(device)
    n1, sample_dim = data1.size()
    n2 = data.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5
    lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam**0.5), steps=n_l1))
    mswd_l0, V_opt, _ = maxSlicedWDL0_L1Approx(data1, data2, p1, p2, lam, lam_l1, candidate_adaptive=candidate_adaptive, n_l1=n_l1, reps=reps)
    Tn = scale * mswd_l0 
    # print('mswd test statistic {}'.format(Tn))
    mswd_thresh, mswd_boots_sample = maxSlicedWDL0_L1Approx_bootstrap(data1, data2, lam, lam_l1, candidate_adaptive=candidate_adaptive, n_l1=n_l1, reps=1, B=B, alpha=alpha)
    mswd_pval = torch.mean((mswd_boots_sample>Tn).float())
    mswd_decision = (Tn>mswd_thresh)
    # mswd_decision = (mswd_pval < alpha)
    V_opt = V_opt.to('cpu')
    return {'statistic': Tn, 'p-value': mswd_pval, 'decision': mswd_decision, 'projection': V_opt, 'bootstrap statistic': mswd_boots_sample}

def is_signif_direction(data1, data2, V, bs, alpha=0.05):
    # data1, data2: tensor
    # V: tensor, given directions of interest; bs: tensor, bootstrap statistics
    data1 = data1.to('cpu')
    data2 = data2.to('cpu')
    n1, sample_dim = data1.size()
    n2 = data.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5
    data_proj1 = torch.matmul(data1, V)
    data_proj2 = torch.matmul(data2, V)
    num_directions = data_proj1.size(1)
    thresh = torch.quantile(bs, 1-alpha)
    statistic = torch.zeros(num_directions)
    pvals = torch.zeros(num_directions)
    decision = torch.zeros(num_directions)
    for i in range(num_directions):
        statistic[i] = scale * wasserstein_distance(data_proj1[:,i], data_proj2[:,i], p1, p2)
        pvals[i] = torch.mean((bs>statistic[i]).float())
        # decision[i] = (pvals[i] < alpha)
        decision[i] = (statistic[i] > thresh)
    
    return {'statistic': statistic, 'p-value': pvals, 'decision': decision}
    

def is_signif_variables(data1, data2, idx, lam, bs, alpha=0.05, candidate_adaptive=True, n_l1=10, reps=10):
    # data1, data2: tensor
    # idx: numpy array, index of variables of interest; bs: tensor, bootstrap statistic
    data1 = data1.to(device)
    data2 = data2.to(device)
    n1, sample_dim = data1.size()
    n2 = data2.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5
    num = idx.size
    data_sub1 = data1[:, idx]
    data_sub2 = data2[:, idx]
    lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam**0.5), steps=n_l1))
    max_wd, V_opt, _ = maxSlicedWDL0_L1Approx(data_sub1, data_sub2, p1, p2, lam, lam_l1, candidate_adaptive=candidate_adaptive, n_l1=n_l1, reps=reps)     
    statistic = scale * max_wd
    pval = torch.mean((bs>statistic).float())
    # decision = (pval < alpha)
    thresh = torch.quantile(bs, 1-alpha)
    decision = (statistic > thresh)
    V_opt = V_opt.to('cpu')
    V = torch.zeros(sample_dim,1)
    V[idx,:] = V_opt
    return {'statistic': statistic, 'decision': decision, 'p-value': pval, 'direction': V}



