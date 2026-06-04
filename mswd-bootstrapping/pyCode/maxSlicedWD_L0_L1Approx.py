import torch
import numpy as np
from scipy.stats import wasserstein_distance
from pyCode.maxSlicedWD_L1 import maxSlicedWD
from pyCode.myProj import myProj
from torch.distributions.multinomial import Multinomial
# from numpy.random import multinomial
from pyCode.global_var import device

def tune_l0(X, Y, lams, k=2, reps=10, candidate_adaptive=True, n_l1=10):
    X = X.to(device)
    Y = Y.to(device)
    n_lam = lams.size(0)
    n1 = X.size(0)
    n2 = Y.size(0)
    dim = X.size(1)
    n1_test = n1 // k
    n2_test = n2 // k
    n1_train = n1 - n1_test
    n2_train = n2 - n2_test
    rec_mswd = torch.zeros(k, n_lam)
    print('cross validation: sparsity parameter')
    for i in range(k):
        if n1==n2:
            loc = np.linspace(n1_test*i, n1_test*(i+1), num=n1_test, endpoint=False, dtype=int)
            X_te = X[loc,:]
            Y_te = Y[loc,:]
            loc2 = np.delete(np.arange(n1), loc)
            X_tr = X[loc2,:]
            Y_tr = Y[loc2,:]
        else:
            loc = np.linspace(n1_test*i, n1_test*(i+1), num=n1_test, endpoint=False, dtype=int)
            X_te = X[loc,:]
            loc2 = np.delete(np.arange(n1), loc)
            X_tr = X[loc2,:]
            loc = np.linspace(n2_test*i, n2_test*(i+1), num=n2_test, endpoint=False, dtype=int)
            Y_te = Y[loc,:]                
            loc2 = np.delete(np.arange(n2), loc)
            Y_tr = Y[loc2,:]
        for i_lam in range(n_lam):
            p1 = torch.ones(n1_train)/n1_train
            p2 = torch.ones(n2_train)/n2_train
            p1_te = torch.ones(n1_test)/n1_test
            p2_te = torch.ones(n2_test)/n2_test            
            # print('{} fold, lam id {}'.format(i, i_lam))
            lam = lams[i_lam]
            lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam), steps=n_l1))  
            _, V_opt, _ = maxSlicedWDL0_L1Approx(X_tr, Y_tr, p1, p2, lam, lam_l1, candidate_adaptive=candidate_adaptive, n_l1=n_l1, reps=reps)
            data_proj1 = torch.matmul(X_te, V_opt).squeeze().to('cpu')
            data_proj2 = torch.matmul(Y_te, V_opt).squeeze().to('cpu')
            rec_mswd[i, i_lam] = wasserstein_distance(data_proj1, data_proj2, p1_te, p2_te)
    lam = lams[torch.argmax(torch.mean(rec_mswd, dim=0))]
    return lam

def maxSlicedWDL0_L1Approx(data1, data2, p1, p2, lam_l0, lam_l1, candidate_adaptive=True, n_l1=10, reps=10, learn_rate=100, thresh=1e-6):
    """
        optimize the max-sliced Wasserstein distance based on the quantile function under l0 sparsity
        use l1 sparsity optimization algorithm to approximate the solution
        lam_l0: l0 sparsity parameter
        lam_l1: l1 sparsity parameters
    """
    data1 = data1.to(device)
    data2 = data2.to(device)
    n1, sample_dim = data1.size()
    n2 = data2.size(0)
    p1 = p1.to(device)
    p2 = p2.to(device)
    if candidate_adaptive == True:
        nonzeros = sample_dim
        lam_l1 = lam_l0**0.5
        lam_l1_seq = []
        lam_l1_seq.append(lam_l1)
        while nonzeros > lam_l0:
            max_mswd = -1
            mswd = torch.zeros(reps)
            for i in range(reps): # different initial values to alleviate the non-convexity issue
                V0 = torch.randn(sample_dim, 1, device=device)
                V0 = V0 / torch.norm(V0)
                V0 = myProj(V0, lam_l1)
                mswd[i], V = maxSlicedWD(data1, data2, V0, p1, p2, lam=lam_l1, learn_rate=learn_rate, thresh=thresh)
                if mswd[i] > max_mswd:
                    V_opt = V.clone()
                    V0_opt = V0.clone()
                    max_mswd = mswd[i]
            nonzeros = torch.norm(V_opt, 0)
            lam_l1 = (1+lam_l1)/2
            lam_l1_seq.append(lam_l1)
        if len(lam_l1_seq)>=2:    
            max_lam_l1 = lam_l1_seq[len(lam_l1_seq)-2]
            lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(max_lam_l1), steps=n_l1))  
        else:
            max_lam_l1 = lam_l1_seq[0]
            lam_l1 = torch.tensor([sample_dim**0.5])             
    m = lam_l1.size(0)
    nonzeros = torch.zeros(m)
    mswd_seq = torch.zeros(m)
    loc_seq = {}
    for j in range(m):
        lam = lam_l1[j]
        max_mswd = -1
        mswd = torch.zeros(reps)
        for i in range(reps): # different initial values to alleviate the non-convexity issue
            V0 = torch.randn(sample_dim, 1, device=device)
            V0 = V0 / torch.norm(V0)
            V0 = myProj(V0, lam)
            mswd[i], V = maxSlicedWD(data1, data2, V0, p1, p2, lam=lam, learn_rate=learn_rate, thresh=thresh)
            if mswd[i] > max_mswd:
                V_opt = V.clone()
                V0_opt = V0.clone()
                max_mswd = mswd[i]
        nonzeros[j] = torch.norm(V_opt, 0)
        loc_seq[j] = torch.where(V_opt.squeeze()!=0)[0]
        mswd_seq[j] = max_mswd
    loc = torch.where(nonzeros<=lam_l0)[0]
    loc = loc[torch.argmax(mswd_seq[loc])]
    lam_l1_selected = lam_l1[loc]
    loc_nonzeros = loc_seq[int(loc)]
    nonzeros = nonzeros[loc]
    if nonzeros==1:
        V = torch.zeros(sample_dim,1, device=device)
        V[loc_nonzeros,0] = 1
        mswd_l0 = mswd_seq[loc]
    else:
        V0 = torch.randn(int(nonzeros),1,device=device)
        V0 = V0/torch.norm(V0)
        mswd_l0, V1 = maxSlicedWD(data1[:,loc_nonzeros], data2[:,loc_nonzeros], V0, p1, p2, lam=nonzeros**0.5, learn_rate=learn_rate, thresh=thresh)
        V = torch.zeros(sample_dim, 1, device=device)
        V[loc_nonzeros,:] = V1
    return mswd_l0, V, lam_l1_selected

def maxSlicedWDL0_L1Approx_bootstrap(data1, data2, lam_l0, lam_l1, candidate_adaptive=True, n_l1=10, reps=1, B=500, alpha=0.05, learn_rate=100, thresh=1e-6):
    data1 = data1.to(device)
    data2 = data2.to(device)
    n1, sample_dim = data1.size()
    n2 = data2.size(0)
    data = torch.cat((data1, data2), axis=0)
    n = n1 + n2
    mswd = torch.zeros(B)
    for i_bootstrap in range(B):
        gen1 = Multinomial(n1, torch.ones(n1, device=device)/n1) # generate independent multinomial samples (n1, 1/n1)
        epsilon1 = gen1.sample()
        gen2 = Multinomial(n2, torch.ones(n2, device=device)/n2) # generate independent multinomial samples (n2, 1/n2)
        epsilon2 = gen2.sample()
        p1 = torch.cat((epsilon1/(2*n1),torch.ones(n2,device=device)/(2*n2)))
        p2 = torch.cat((torch.ones(n1,device=device)/(2*n1), epsilon2/(2*n2)))
        # epsilon1 = multinomial(n1, np.ones(n1)/n1)
        # epsilon2 = multinomial(n2, np.ones(n2)/n2)
        # epsilon1 = torch.from_numpy(epsilon1)
        # epsilon2 = torch.from_numpy(epsilon2)
        # p1 = torch.cat((epsilon1/(2*n1),torch.ones(n2)/(2*n2)))
        # p2 = torch.cat((torch.ones(n1)/(2*n1), epsilon2/(2*n2)))
        mswd[i_bootstrap],_,_ = maxSlicedWDL0_L1Approx(data, data, p1, p2, lam_l0=lam_l0, lam_l1=lam_l1, candidate_adaptive=candidate_adaptive, n_l1=n_l1, reps=reps, learn_rate=learn_rate, thresh=thresh)
        if (i_bootstrap+1)%B==0:
            print('max sliced wd empirical bootstrap {} thresh {}'.format(i_bootstrap+1, 2*(n1*n2/(n1+n2))**0.5*torch.quantile(mswd[0:(i_bootstrap+1)], q=1-alpha)))    
    mswd = 2* mswd * (n1*n2/(n1+n2))**0.5
    mswd_bootstrap_thresh = torch.quantile(mswd, q=1-alpha)
    return mswd_bootstrap_thresh, mswd