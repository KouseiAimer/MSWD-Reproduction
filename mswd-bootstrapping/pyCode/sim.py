import torch
import numpy as np
# import ot
# from scipy.stats import wasserstein_distance
from pyCode.permTest import permTest
from pyCode.maxSlicedWD_L1 import maxSlicedWD, maxSlicedWD_bootstrap, tune_l1
from pyCode import myProj
from pyCode.sMMD import sMMD_test
from pyCode.pwdPerm import pwdPerm
from pyCode.WD_NNapprox import get_WDapprox, WDapprox_bootstrap
from pyCode.global_var import device

def sim(data1, data2, opts):
    """
        function to run different methods in a single file
        signal: tensor
        methods: dir {'perm': True, 'mswd_l1':True, 'smmd': True} # implement the method or not
        permutation methods include 'mmd', 'ed_l1', 'ed_l2', 'bg'    
    """
    alpha = opts.alpha
    data1 = data1.to(device)
    data2 = data2.to(device)
    n1, sample_dim = data1.size()
    n2 = data2.size(0)

    results = {}
    if opts.methods['mswd_l1']==True:
        if hasattr(opts, 'nB'):
            nB = opts.nB
        else:
            nB = 500
        p1 = torch.ones(n1)/n1
        p2 = torch.ones(n2)/n2
        if hasattr(opts, 'tune'):
            if opts.tune:
                lams = opts.lams
                lam = tune_l1(data1, data2, lams, k=2, reps=5)
            else:
                lam = sample_dim**0.5 # no sparsity
        elif hasattr(opts, 'lam'):
            lam = opts.lam
        else:
            lam = sample_dim**0.5
        if hasattr(opts, 'reps'):
            reps = opts.reps
        else:
            reps=1
        scale = (n1*n2/(n1+n2))**0.5
        max_mswd = 0
        mswd = torch.zeros(reps)
        for i in range(reps): # different initial values to alleviate the non-convexity issue
            # V0 = torch.randn(sample_dim, 1)
            V0 = np.random.normal(0,1,(sample_dim,1))
            V0 = torch.from_numpy(V0).float()
            V0 = V0 / torch.norm(V0)
            V0 = myProj.myProj(V0, lam)
            mswd[i], V = maxSlicedWD(data1, data2, V0, p1, p2, lam=lam, learn_rate=100, thresh=1e-6)
            if mswd[i] > max_mswd:
                V_opt = V.clone()
                V0_opt = V0.clone()
                max_mswd = mswd[i]     
        # print('mswd {}'.format(mswd))
        # print('parameter after tuning {}'.format(lam))
        Tn = scale * max_mswd
        print('mswd l1 test statistic {}'.format(Tn))
        mswd_l1_thresh, mswd_l1_boots_sample = maxSlicedWD_bootstrap(data1, data2, V0_opt, lam=lam, B=nB, alpha=alpha, learn_rate=100, thresh=1e-6)
        mswd_l1_decision = (Tn>mswd_l1_thresh)
        mswd_l1_pval = torch.mean((mswd_l1_boots_sample>Tn).float())
        results['mswd_l1_decision'] = mswd_l1_decision
        results['mswd_l1_pval'] = mswd_l1_pval
        results['l1_projection'] = V_opt
    
    if opts.methods['perm']==True:
        perm = permTest(data1, data2, alpha=alpha)
        results['perm'] = perm
    
    if opts.methods['smmd']==True:
        smmd = sMMD_test(data1,data2, alpha=alpha)
        results['smmd_decision'] = smmd['smmd_decision']
        results['smmd_pval'] = smmd['smmd_pval']

    if opts.methods['pwdPerm']==True:
        kPWD_perm = pwdPerm(data1, data2, alpha=alpha)
        results['kPWD_perm_decision'] = kPWD_perm['kPWD_perm_decision']
        results['kPWD_perm_pval'] = kPWD_perm['kPWD_perm_pval']

    if opts.methods['wd_NNapprox']==True:

        Tn = get_WDapprox(data1, data2, opts.hyper_params)
        # multiplier bootstrap
        r = torch.linspace(0.01, 0.99, steps=100)
        quantiles1 = (n2/(n1+n2))**0.5 * WDapprox_bootstrap(data1, opts.hyper_params, r, alpha=alpha)
        quantiles2 = (n1/(n1+n2))**0.5 * WDapprox_bootstrap(data2, opts.hyper_params, 1-r, alpha=alpha)
        quantiles = torch.min(quantiles1 + quantiles2)
        wd_nnapprox_decision = (Tn > quantiles)
        results['wd_nnapprox_decision'] = wd_nnapprox_decision

    return results
