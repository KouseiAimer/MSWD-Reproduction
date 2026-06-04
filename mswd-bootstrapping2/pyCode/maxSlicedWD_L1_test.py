
## implement the proposed test given data 
import torch
import numpy as np
from pyCode.maxSlicedWD_L1 import maxSlicedWD, maxSlicedWD_bootstrap
from pyCode.myProj import myProj
from pyCode.global_var import device

def mswdTest(data1, data2, lam, alpha=0.05, B=500, reps=10):
    # data1, data2: tensor
    # lam: sparsity parameter
    # alpha: significance level; B: the number of bootstrapping samples
    # reps: the number of different initial values for optimization

    data1 = data1.to(device)
    data2 = data2.to(device)
    n1, sample_dim = data1.size()
    n2 = data.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5
    max_mswd = 0
    mswd = torch.zeros(reps)
    for i in range(reps): # different initial values to alleviate the non-convexity issue
        V0 = torch.randn(sample_dim, 1)
        V0 = V0 / torch.norm(V0)
        V0 = myProj(V0, lam)
        mswd[i], V = maxSlicedWD(data1, data2, V0, p1, p2, lam=lam, learn_rate=100, thresh=1e-6)
        if mswd[i] > max_mswd:
            V_opt = V.clone()
            V0_opt = V0.clone()
            max_mswd = mswd[i]     
    Tn = scale * max_mswd
    # print('mswd test statistic {}'.format(Tn))
    mswd_thresh, mswd_boots_sample = maxSlicedWD_bootstrap(data1, data2, V0_opt, lam=lam, B=B, alpha=alpha, learn_rate=100, thresh=1e-6)
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
    

def is_signif_variables(data1, data2, idx, lam, bs, alpha=0.05, reps=10):
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
    wd_val = torch.zeros(reps)
    max_wd = 0
    V_opt = torch.ones(num)
    for i in range(reps):
        V0 = torch.randn(num,1)
        V0, _ = torch.linalg.qr(V0)
        wd_val[i], V = maxSlicedWD(data_sub1, data_sub2, V0, p1, p2, lam=lam, learn_rate=100, thresh=1e-6)
        if wd_val[i]>max_wd:
            max_wd = wd_val[i]
            V_opt = V      
    statistic = scale * max_wd
    pval = torch.mean((bs>statistic).float())
    # decision = (pval < alpha)
    thresh = torch.quantile(bs, 1-alpha)
    decision = (statistic > thresh)
    V_opt = V_opt.to('cpu')
    V = torch.zeros(sample_dim,1)
    V[idx,:] = V_opt
    return {'statistic': statistic, 'decision': decision, 'p-value': pval, 'direction': V}



