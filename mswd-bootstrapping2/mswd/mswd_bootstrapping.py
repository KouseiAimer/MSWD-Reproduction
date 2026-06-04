import torch
import numpy as np
import ot
from scipy.stats import wasserstein_distance
from numpy.random import multinomial
from torch.distributions.multinomial import Multinomial

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Two-Sample Distribution Tests for High-Dimensional Data
# @description Test whether the two populations have equal distribution or not
# @param X (Y) tensor n1*p (n2*p) matrix, with each row representing an observation
# @param lam tensor, a candidate set of sparsity parameter(s)
# @param B the number of bootstrap replicates; default value: 500
# @param alpha the significance level; default value: 0.05
# @param reps the number of different initial directions used; default value: 10
# @param istune whether to use k-fold cross validation to tune the sparsity parameter; default value: True
# @param candidate_adaptive whether the sequence of l1 sparsity parameters is adaptively determined; default value: True
# @param n_l1 the number of the sequence of l1 sparsity parameters; default value: 10
# @param tune_k the number of folds for cross validation; default value: 2
# @param tune_reps the number of different initial directions used for cross validation; default value: 5
# @param opt_lr the learning rate for the projected gradient descent algorithm; default value: 100
# @param opt_thresh the tolerance level for the projected gradient descent algorithm; default value: 1e-6
# @param opt_maxIter the maximum number of iterations for the projected gradient descent algorithm; default value: 100
# @param proj_thresh the tolerance level for the quadratic approximation secant bisection method; default value: 1e-6
# @param proj_maxIter the maximum number of iterations for the quadratic approximation secant bisection method; default value: 100
# @return a dictionary with
#               \key{test statistic}: tensor test statistic
#               \key{pvalue}: tensor p-value
#               \key{reject}: 1 (True) or 0 (False), whether the null hypothesis is rejected
#               \key{accept}: 1 (True) or 0 (False), whether the null hypothesis is not rejected
#               \key{optimal direction}: tensor, the optimal projection direction that maximizes the Wasserstein distance between the resulting univariate distributions
#               \key{alpha}: float, the significance level
#               \key{bootstrap sample}: tensor, the bootstrapped statistic
#               \key{lam}: tensor, the sparsity parameter(s)
#               \key{selected lam}: the selected sparsity parameter with k-fold cross validation
#               \key{scale}: (n1*n2/(n1+n2))**0.5
#               \key{sci bound optproj}: the bound of the SCI for the max-sliced Wasserstein distance (under the optimal projection direction)
#               \key{selected initial direction}: tensor, the initial direction used for obtaining the final optimal direction

def mswdtest(X, Y, lam, istune=True, reps=10, tune_k=2, tune_reps=5, candidate_adaptive=True, n_l1=10, alpha=0.05, B=500, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
    X = X.to(device)
    Y = Y.to(device)
    n1, sample_dim = X.size()
    n2 = Y.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5

    if istune == True:
        if lam.size() == torch.Size([]):
            lam = torch.tensor([lam])
        if lam.size(0) > 1:
            lam_selected = tune_l0(X, Y, lam, k=tune_k, reps=tune_reps)
        else:
            lam_selected = lam[0]
        lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam_selected**0.5), steps=n_l1))  
        mswd, V, _ = maxSlicedWDL0_L1Approx(X, Y, p1, p2, lam_selected, lam_l1, candidate_adaptive, n_l1, reps, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
        Tn = scale * mswd
        _, mswd_bsample = maxSlicedWDL0_L1Approx_bootstrap(X, Y, lam_selected, lam_l1, candidate_adaptive, n_l1, reps, B, alpha, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
        pval = torch.mean((mswd_bsample>Tn).float())
        reject = (pval < alpha)
        accept = (pval >= alpha)
        sci_bound_optproj = (Tn - torch.quantile(mswd_bsample, 1-alpha)) / scale
        return {'test statistic': Tn, 'reject': reject, 'accept': accept, 'pvalue': pval, 'optimal direction': V, 'alpha': alpha, \
                'bootstrap sample': mswd_bsample, 'lam': lam, 'selected lam': lam_selected, 'sci bound optproj': sci_bound_optproj, \
                'scale': scale}

    else:
        nlam = lam.size(0)
        Tn = torch.zeros(nlam)
        pval = torch.zeros(nlam)
        reject = torch.zeros(nlam)
        accept = torch.zeros(nlam)
        sci_bound_optproj = torch.zeros(nlam)
        V_all = torch.zeros(sample_dim, nlam)
        V0_all = torch.zeros(sample_dim, nlam)
        mswd_bsample_all = torch.zeros(B, nlam)

        for i in range(nlam):
            lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam[i]**0.5), steps=n_l1))  
            mswd, V, _ = maxSlicedWDL0_L1Approx(X, Y, p1, p2, lam[i], lam_l1, candidate_adaptive, n_l1, reps, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
            Tn[i] = scale * mswd
            V_all[:, i:(i+1)] = V
            _, mswd_bsample = maxSlicedWDL0_L1Approx_bootstrap(X, Y, lam[i], lam_l1, candidate_adaptive, n_l1, reps, B, alpha, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
            mswd_bsample_all[:, i] = mswd_bsample
            pval[i] = torch.mean((mswd_bsample>Tn[i]).float())
            reject[i] = (pval[i] < alpha)
            accept[i] = (pval[i] >= alpha)
            sci_bound_optproj[i] = (Tn[i] - torch.quantile(mswd_bsample, 1-alpha)) / scale
        return {'test statistic': Tn, 'reject': reject, 'accept': accept, 'pvalue': pval, 'optimal direction': V_all, \
                    'bootstrap sample': mswd_bsample_all, 'alpha': alpha, 'lam': lam, 'sci bound optproj': sci_bound_optproj, \
                    'scale': scale}

# @description Construct the one-sided simultaneous confidence interval (SCI) for projection directions of interest
# @param V tensor, the projection directions of interest
# @param bsample tensor, the bootstrapped statistic
# @details check if the projection direction satisfy \|v\|_1 <= lam, if so, return the corresponding results,
#          if not, return nan
def mswd_sci_direction(X, Y, V, lam, bsample=None, candidate_adaptive=True, n_l1=10, B=500, reps=10, tune_k=2, tune_reps=5, alpha=0.05, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
    
    if lam.size() == torch.Size([]):
        lam = torch.tensor([lam])
    if lam.size(0) > 1:
        lam_selected = tune_l0(X, Y, lam, k=tune_k, reps=tune_reps)
        # if the selected sparsity parameter is not provided
        # a candidate set of sparsity parameters is given, then we should obtain the bootstrapped
        # sample with the selected sparsity parameter by cross validation
        bsample = None
    else:
        lam_selected = lam[0]

    X = X.to(device)
    Y = Y.to(device)
    V = V.to(device)
    n1, sample_dim = X.size()
    n2 = Y.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5
    if bsample == None:
        lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam_selected**0.5), steps=n_l1))  
        _, bsample = maxSlicedWDL0_L1Approx_bootstrap(X, Y, lam_selected, lam_l1, candidate_adaptive, n_l1, reps, B, alpha, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
    
    thresh = torch.quantile(bsample, 1-alpha)
    num_directions = V.size(1)
    X_proj = torch.matmul(X, V).to('cpu')
    Y_proj = torch.matmul(Y, V).to('cpu')
    num_directions = X_proj.size(1)
    distance = torch.zeros(num_directions)
    sci_bounds = torch.zeros(num_directions)
    for i in range(num_directions):
        if torch.norm(V[:,i:(i+1)], 0) > lam_selected:
            distance[i] = float('nan')
            sci_bounds[i] = float('nan')
        else:
            distance[i] = wasserstein_distance(X_proj[:,i], Y_proj[:,i], p1, p2)
            sci_bounds[i] = distance[i] - thresh / scale
    V = V.to('cpu')
    return {'sci bounds': sci_bounds, 'distance': distance, 'lam': lam, 'selected lam': lam_selected, \
            'bootstrap sample': bsample, 'alpha': alpha, 'direction': V, 'scale': scale}

# @description Construct the one-sided simultaneous confidence interval (SCI) for marginal distributions of interest 
# @param idx list, the indices of marginal coordinates of interest
# @detail given a subset of coordinates, compute the max-sliced Wasserstein distance on this subset
def mswd_sci_marginal(X, Y, idx, lam, bsample=None, B=500, reps=10, tune_k=2, tune_reps=5, alpha=0.05, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
    
    if lam.size() == torch.Size([]):
        lam = torch.tensor([lam])
    if lam.size(0) > 1:
        lam_selected = tune_l0(X, Y, lam, k=tune_k, reps=tune_reps)
        # if the selected sparsity parameter is not provided
        # a candidate set of sparsity parameters is given, then we should obtain the bootstrapped
        # sample with the selected sparsity parameter by cross validation
        bsample = None
    else:
        lam_selected = lam[0]

    X = X.to(device)
    Y = Y.to(device)
    n1, sample_dim = X.size()
    n2 = Y.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    scale = (n1*n2/(n1+n2))**0.5
    if bsample == None:
        lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam_selected**0.5), steps=n_l1))  
        _, bsample = maxSlicedWDL0_L1Approx_bootstrap(X, Y, lam_selected, lam_l1, candidate_adaptive, n_l1, reps, B, alpha, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
    
    thresh = torch.quantile(bsample, 1-alpha)
    num_marginals = len(idx)
    lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam_selected**0.5), steps=n_l1))  
    distance = torch.zeros(num_marginals)
    sci_bounds = torch.zeros(num_marginals)
    # directions = [None] * num_marginals # list
    directions = torch.zeros(sample_dim, num_marginals)
    for i in range(num_marginals):
        X_tmp = X[:, idx[i]]
        Y_tmp = Y[:, idx[i]]
        mswd, V, _ = maxSlicedWDL0_L1Approx(X_tmp, Y_tmp, p1, p2, lam_selected, lam_l1, candidate_adaptive, n_l1, reps, opt_lr, opt_thresh, opt_maxIter, proj_thresh, proj_maxIter)
        directions[idx[i],i:(i+1)] = V
        sci_bounds[i] = distance[i] - thresh / scale
    return {'sci bounds': sci_bounds, 'distance': distance, 'optimal direction': directions, 'bootstrap sample': bsample, \
            'alpha': alpha, 'lam': lam, 'selected lam': lam_selected, 'index set': idx, 'scale': scale}


def get_indices_quantile(xv_indices, q1, q):
    """
        get the corresponding indices in the original data (xv) for the quantiles (q)
        xv_indices: xv[xv_indices] = xv_sorted
        q1: CDF of xv_sorted
    """
    n = q1.size(0)
    nn = q.size(0)
    q1_mat = q1.reshape(n,1).repeat(1,nn)
    q_mat = q.reshape(1,nn).repeat(n,1)
    upper = torch.where(q1_mat>=q_mat, 1, 0)
    x_indices = xv_indices[torch.argmax(upper,0)]
    return x_indices
    
def get_wd(x, y, p1, p2):
    """
        obtain the 1-Wasserstein distance between univariate distributions based on quantiles
    """
    x = x.to(device)
    y = y.to(device)
    x = x.squeeze()
    y = y.squeeze()
    p1 = p1.to(device)
    p2 = p2.to(device)
    n1 = x.size(0)
    n2 = y.size(0)
    x_ordered, x_indices = torch.sort(x)
    q1 = torch.cumsum(p1[x_indices], dim=0)
    y_ordered, y_indices = torch.sort(y)
    q2 = torch.cumsum(p2[y_indices], dim=0)
    q = torch.unique(torch.cat((q1,q2)))
    q, _ = torch.sort(q)
    nn = q.size(0)
    x_indices = get_indices_quantile(x_indices, q1, q)
    y_indices = get_indices_quantile(y_indices, q2, q)
    # x_indices = x_indices[torch.ceil(n1*q).long()-1]
    # y_indices = y_indices[torch.ceil(n2*q).long()-1] # it may cause issues due to computation accuracy
    # e.g., n1*q should be an integer, but it turns out to be a bit larger than the integer
    # after torch.ceil(), it is larger than the orignal order by 1.
    dq = q - torch.cat((torch.zeros(1, device=device), q[0:(nn-1)]))
    wd = torch.sum(torch.abs(x[x_indices] - y[y_indices]) * dq)
    return wd


def get_subgrad_quantile(x, y, v, p1, p2):
    """
        get the subgradient of the W_1(v^\t x, v^\t y) with respect to v based on the quantile function
        probability mass p1 and p2
    """
    x = x.to(device)
    y = y.to(device)
    v = v.to(device)
    n1 = x.size(0)
    n2 = y.size(0)
    sample_dim = x.size(1)
    xv = torch.matmul(x, v).squeeze()
    yv = torch.matmul(y, v).squeeze()
    xv_ordered, xv_indices = torch.sort(xv)
    q1 = torch.cumsum(p1[xv_indices], dim=0)
    yv_ordered, yv_indices = torch.sort(yv)
    q2 = torch.cumsum(p2[yv_indices], dim=0)
    q = torch.unique(torch.cat((q1,q2)))
    q, _ = torch.sort(q)
    nn = q.size(0)
    x_indices = get_indices_quantile(xv_indices, q1, q)
    y_indices = get_indices_quantile(yv_indices, q2, q)
    dq = q - torch.cat((torch.zeros(1,device=device), q[0:(nn-1)]))
    sign_deltaq = torch.diag(torch.sign(xv[x_indices] - yv[y_indices]) * dq)
    subgrad = torch.sum(torch.matmul(sign_deltaq, x[x_indices,:] - y[y_indices,:]), 0)
    subgrad = subgrad.reshape(sample_dim,1)
    return subgrad


def maxSlicedWD_L1(x, y, v0, p1, p2, lam, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
    """
        optimize the max-sliced Wasserstein distance based on the quantile function
        output distance and the optimal projection v
        v0: initial value of v
        p1, p2: marginal weights
        lam: sparsity parameter, which is greater than or equal to 1
    """
    x = x.to(device)
    y = y.to(device)
    v0 = v0.to(device)
    p1 = p1.to(device)
    p2 = p2.to(device)
    old_v = v0.clone()
    iter_id = 0
    error = 1
    while error > opt_thresh and iter_id <= opt_maxIter:
        iter_id = iter_id+1   
        opt_lr = opt_lr / iter_id**0.5
        # opt_lr = 0.8 * opt_lr
        # opt_lr = 1/iter_id**0.5 
        subgrad = get_subgrad_quantile(x, y, old_v, p1, p2)
        v = old_v + opt_lr*subgrad
        v = myProj(v, lam, proj_thresh, proj_maxIter)
        error = torch.linalg.norm(v-old_v) / (1+torch.linalg.norm(old_v))
        old_v = v.clone()
        # print('the iteration {}, learn rate {}, the error {}'.format(iter_id, opt_lr, error))
        # print('the first few coordinates {}'.format(old_v[0:10]))
    x_proj = torch.matmul(x, v).squeeze()
    y_proj = torch.matmul(y, v).squeeze()
    mswd = wasserstein_distance(x_proj.to('cpu'), y_proj.to('cpu'), p1.to('cpu'), p2.to('cpu'))
    mswd = torch.tensor(mswd).float() 
    return mswd, v
    
def tune_l0(X, Y, lams, k=2, reps=10, candidate_adaptive=True, n_l1=10, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
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
            _, V_opt, _ = maxSlicedWDL0_L1Approx(X_tr, Y_tr, p1, p2, lam, lam_l1, candidate_adaptive=candidate_adaptive, n_l1=n_l1, reps=reps, opt_lr=opt_lr, opt_thresh=opt_thresh, opt_maxIter=opt_maxIter, proj_thresh=proj_thresh, proj_maxIter=proj_maxIter)
            data_proj1 = torch.matmul(X_te, V_opt).squeeze().to('cpu')
            data_proj2 = torch.matmul(Y_te, V_opt).squeeze().to('cpu')
            rec_mswd[i, i_lam] = wasserstein_distance(data_proj1, data_proj2, p1_te, p2_te)
    lam = lams[torch.argmax(torch.mean(rec_mswd, dim=0))]
    return lam

def maxSlicedWDL0_L1Approx(data1, data2, p1, p2, lam_l0, lam_l1, candidate_adaptive=True, n_l1=10, reps=10, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
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
                V0 = myProj(V0, lam_l1, proj_thresh, proj_maxIter)
                mswd[i], V = maxSlicedWD_L1(data1, data2, V0, p1, p2, lam=lam_l1, opt_lr=opt_lr, opt_thresh=opt_thresh, opt_maxIter=opt_maxIter, proj_thresh=proj_thresh, proj_maxIter=proj_maxIter)
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
            V0 = myProj(V0, lam, proj_thresh, proj_maxIter)
            mswd[i], V = maxSlicedWD_L1(data1, data2, V0, p1, p2, lam=lam, opt_lr=opt_lr, opt_thresh=opt_thresh, opt_maxIter=opt_maxIter, proj_thresh=proj_thresh, proj_maxIter=proj_maxIter)
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
        mswd_l0, V1 = maxSlicedWD_L1(data1[:,loc_nonzeros], data2[:,loc_nonzeros], V0, p1, p2, lam=nonzeros**0.5, opt_lr=opt_lr, opt_thresh=opt_thresh, opt_maxIter=opt_maxIter, proj_thresh=proj_thresh, proj_maxIter=proj_maxIter)
        V = torch.zeros(sample_dim, 1, device=device)
        V[loc_nonzeros,:] = V1
    return mswd_l0, V, lam_l1_selected

def maxSlicedWDL0_L1Approx_bootstrap(data1, data2, lam_l0, lam_l1, candidate_adaptive=True, n_l1=10, reps=1, B=500, alpha=0.05, opt_lr=100, opt_thresh=1e-6, opt_maxIter=100, proj_thresh=1e-6, proj_maxIter=100):
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
        mswd[i_bootstrap],_,_ = maxSlicedWDL0_L1Approx(data, data, p1, p2, lam_l0=lam_l0, lam_l1=lam_l1, candidate_adaptive=candidate_adaptive, n_l1=n_l1, reps=reps, opt_lr=opt_lr, opt_thresh=opt_thresh, opt_maxIter=opt_maxIter, proj_thresh=proj_thresh, proj_maxIter=proj_maxIter)
        if (i_bootstrap+1)%B==0:
            print('max sliced wd empirical bootstrap {} thresh {}'.format(i_bootstrap+1, 2*(n1*n2/(n1+n2))**0.5*torch.quantile(mswd[0:(i_bootstrap+1)], q=1-alpha)))    
    mswd = 2* mswd * (n1*n2/(n1+n2))**0.5
    mswd_bootstrap_thresh = torch.quantile(mswd, q=1-alpha)
    return mswd_bootstrap_thresh, mswd


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

def myProj(V, lam, proj_thresh=1e-6, proj_maxIter=100):
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

