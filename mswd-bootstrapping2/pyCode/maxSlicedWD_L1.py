import torch
import numpy as np
from scipy.stats import wasserstein_distance
from pyCode import myProj
# from torch.distributions.multinomial import Multinomial
from numpy.random import multinomial
from pyCode.global_var import device


def tune_l1(X, Y, lams, reps=2, k=5):
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
            lam = lams[i_lam]
            mswd_max = 0
            for i_rep in range(reps):
                # V0 = torch.randn(dim, 1)
                V0 = np.random.normal(0,1,(dim,1))
                V0 = torch.from_numpy(V0).float()
                V0 = myProj.myProj(V0, lam)
                mswd, V = maxSlicedWD(X_tr, Y_tr, V0, p1, p2, lam)
                if mswd > mswd_max:
                    V_opt = V.clone()
                    mswd_max = mswd.clone()
            data_proj1 = torch.matmul(X_te, V_opt).squeeze().to('cpu')
            data_proj2 = torch.matmul(Y_te, V_opt).squeeze().to('cpu')
            rec_mswd[i, i_lam] = wasserstein_distance(data_proj1, data_proj2, p1_te, p2_te)
    lam = lams[torch.argmax(torch.mean(rec_mswd, dim=0))]
    return lam


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


def maxSlicedWD(x, y, v0, p1, p2, lam, learn_rate=100, thresh=1e-6, max_iter=100):
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
    while error > thresh and iter_id <= max_iter:
        iter_id = iter_id+1   
        learn_rate = learn_rate / iter_id**0.5
        # learn_rate = 0.8 * learn_rate
        # learn_rate = 1/iter_id**0.5 
        subgrad = get_subgrad_quantile(x, y, old_v, p1, p2)
        v = old_v + learn_rate*subgrad
        v = myProj.myProj(v, lam)
        error = torch.linalg.norm(v-old_v) / (1+torch.linalg.norm(old_v))
        old_v = v.clone()
        # print('the iteration {}, learn rate {}, the error {}'.format(iter_id, learn_rate, error))
        # print('the first few coordinates {}'.format(old_v[0:10]))
    x_proj = torch.matmul(x, v).squeeze()
    y_proj = torch.matmul(y, v).squeeze()
    mswd = wasserstein_distance(x_proj.to('cpu'), y_proj.to('cpu'), p1.to('cpu'), p2.to('cpu'))
    mswd = torch.tensor(mswd).float() 
    return mswd, v
    

def maxSlicedWD_bootstrap(X, Y, V0, lam, learn_rate=100, thresh=1e-6, B=500, alpha=0.05):
    """
        approximate quantiles by the empirical bootstrap 
        sampling multinomial variables
    """
    X = X.to(device)
    Y = Y.to(device)
    V0 = V0.to(device)
    n1 = X.size(0)
    n2 = Y.size(0)
    data = torch.cat((X, Y), axis=0)
    n = n1 + n2
    mswd = torch.zeros(B)
    for i_bootstrap in range(B):
        # gen1 = Multinomial(n1, torch.ones(n1, device=device)/n1) # generate independent multinomial samples (n1, 1/n1)
        # epsilon1 = gen1.sample()
        # gen2 = Multinomial(n2, torch.ones(n2, device=device)/n2) # generate independent multinomial samples (n2, 1/n2)
        # epsilon2 = gen2.sample()
        # p1 = torch.cat((epsilon1/(2*n1),torch.ones(n2,device=device)/(2*n2)))
        # p2 = torch.cat((torch.ones(n1,device=device)/(2*n1), epsilon2/(2*n2)))
        epsilon1 = multinomial(n1, np.ones(n1)/n1)
        epsilon2 = multinomial(n2, np.ones(n2)/n2)
        epsilon1 = torch.from_numpy(epsilon1)
        epsilon2 = torch.from_numpy(epsilon2)
        p1 = torch.cat((epsilon1/(2*n1),torch.ones(n2)/(2*n2)))
        p2 = torch.cat((torch.ones(n1)/(2*n1), epsilon2/(2*n2)))
        mswd[i_bootstrap],_ = maxSlicedWD(data, data, V0, p1, p2, lam=lam, learn_rate=learn_rate, thresh=thresh)
        if (i_bootstrap+1)%100==0:
            print('max sliced wd empirical bootstrap {} thresh {}'.format(i_bootstrap+1, 2*(n1*n2/(n1+n2))**0.5*torch.quantile(mswd[0:(i_bootstrap+1)], q=1-alpha)))    
    mswd = 2* mswd * (n1*n2/(n1+n2))**0.5
    mswd_bootstrap_thresh = torch.quantile(mswd, q=1-alpha)
    return mswd_bootstrap_thresh, mswd


