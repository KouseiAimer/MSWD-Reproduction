import numpy as np
from sklearn.gaussian_process.kernels import RBF
import torch
from pyCode.MMD import kernelSigma
from pyCode.ED import ED
from pyCode.global_var import device

def permTest(data1, data2, nperm=500, alpha=0.05):
    """
        permutation test for mmd, ED_L1, ED_L2, BG test
    """
    n1 = data1.size(0)
    n2 = data2.size(0)
    n = n1+n2
    # ED L1  
    data1 = data1.to(device)
    data2 = data2.to(device) 
    edl1, dist_mat = ED(data1, data2, p=1) # ED_L1 distance; if cuda available, output is cuda type
    # ED L2
    edl2, dist_mat2 = ED(data1, data2, p=2) # ED_L2 distance
    # BG
    dist11 = dist_mat2[0:n1, 0:n1]
    dist22 = dist_mat2[n1:n, n1:n]
    t_scale2 = torch.sum(dist22)/n2/(n2-1) - torch.sum(dist11)/n1/(n1-1)
    t_scale2 = t_scale2.to('cpu')
    bg = (edl2**2 + t_scale2**2) / 2

    # squared MMD: Gaussian kernel
    sigma = kernelSigma(data1, data2)
    sigma = torch.from_numpy(sigma).to(device)
    # rbf = RBF(sigma/2**0.5)
    # data1 = data1.to('cpu')
    # data2 = data2.to('cpu')
    # ker_mat_1 = rbf(data1)
    # ker_mat_2 = rbf(data2)
    # ker_mat_12 = rbf(data1, data2)
    # ker_mat_1 = torch.from_numpy(ker_mat_1).float().to(device)
    # ker_mat_2 = torch.from_numpy(ker_mat_2).float().to(device)
    # ker_mat_12 = torch.from_numpy(ker_mat_12).float().to(device)
    ker_mat_1 = torch.exp(-torch.cdist(data1, data1)**2/sigma**2)
    ker_mat_2 = torch.exp(-torch.cdist(data2, data2)**2/sigma**2)
    ker_mat_12 = torch.exp(-torch.cdist(data1, data2)**2/sigma**2)
    ker_mat_1 = ker_mat_1 - torch.diag(torch.diag(ker_mat_1))
    ker_mat_2 = ker_mat_2 - torch.diag(torch.diag(ker_mat_2))
    mmd = torch.sum(ker_mat_1)/n1/(n1-1) + torch.sum(ker_mat_2)/n2/(n2-1) - 2 * torch.mean(ker_mat_12)
    mmd = mmd.to('cpu')
    ker_mat1 = torch.cat((ker_mat_1,ker_mat_12), 1)
    ker_mat2 = torch.cat((ker_mat_12.T, ker_mat_2), 1)
    ker_mat = torch.cat((ker_mat1, ker_mat2), 0) # used for permutation

    ## MMD: Laplace kernel
    dist12 = dist_mat2[0:n1, n1:n]
    lapker_mat_1 = torch.exp(-dist11/sigma)
    lapker_mat_2 = torch.exp(-dist22/sigma)
    lapker_mat_12 = torch.exp(-dist12/sigma)
    lapker_mat_1 = lapker_mat_1 - torch.diag(torch.diag(lapker_mat_1))
    lapker_mat_2 = lapker_mat_2 - torch.diag(torch.diag(lapker_mat_2))
    lapmmd = torch.sum(lapker_mat_1)/n1/(n1-1) + torch.sum(lapker_mat_2)/n2/(n2-1) - 2*torch.mean(lapker_mat_12)
    lapmmd = lapmmd.to('cpu')
    lapker_mat1 = torch.cat((lapker_mat_1, lapker_mat_12), 1)
    lapker_mat2 = torch.cat((lapker_mat_12.T, lapker_mat_2), 1)
    lapker_mat = torch.cat((lapker_mat1, lapker_mat2), 0)

    ED_L1_perm = torch.zeros(nperm)
    mmd_perm = torch.zeros(nperm)
    ED_L2_perm = torch.zeros(nperm)
    bg_perm = torch.zeros(nperm)
    lapmmd_perm = torch.zeros(nperm)
    for perm_id in range(nperm):
        if (perm_id+1)%100==0:
            print('run {} of permutation'.format(perm_id+1))
        # loc = np.random.choice(n, n1, replace=False)
        # dist_mat11 = dist_mat[loc, :][:, loc]
        # # loc = loc.reshape((n1,1))
        # # dist_mat11 = dist_mat[loc, loc.transpose()]
        # loc2 = np.delete(np.arange(n), loc)
        # dist_mat22 = dist_mat[loc2, :][:, loc2]
        # dist_mat12 = dist_mat[loc, :][:, loc2]
        locperm = np.random.permutation(n)
        loc = locperm[0:n1]
        loc2 = locperm[n1:n]
        dist_mat11 = dist_mat[loc, :][:, loc]
        dist_mat22 = dist_mat[loc2, :][:, loc2]
        dist_mat12 = dist_mat[loc, :][:, loc2]
        ED_L1_perm[perm_id] = 2 * torch.mean(dist_mat12) - torch.sum(dist_mat11) / n1 / (n1 - 1) - torch.sum(dist_mat22) / n2 / (n2 - 1)

        dist_mat11 = dist_mat2[loc, :][:, loc]
        dist_mat12 = dist_mat2[loc, :][:, loc2]
        dist_mat22 = dist_mat2[loc2,:][:, loc2]
        ED_L2_perm[perm_id] = 2 * torch.mean(dist_mat12) - torch.sum(dist_mat11) / n1 / (n1 - 1) - torch.sum(dist_mat22) / n2 / (n2 - 1)

        t_scale2 = torch.sum(dist_mat22)/n2/(n2-1) - torch.sum(dist_mat11)/n1/(n1-1)
        bg_perm[perm_id] = (ED_L2_perm[perm_id]**2 + t_scale2**2)/2

        ker_mat11 = ker_mat[loc, :][:, loc]
        ker_mat22 = ker_mat[loc2, :][:, loc2]
        ker_mat12 = ker_mat[loc, :][:, loc2]
        mmd_perm[perm_id] = torch.sum(ker_mat11)/n1/(n1-1) + torch.sum(ker_mat22)/n2/(n2-1) - 2 * torch.mean(ker_mat12)

        lapker_mat11 = lapker_mat[loc, :][:, loc]
        lapker_mat22 = lapker_mat[loc2, :][:, loc2]
        lapker_mat12 = lapker_mat[loc, :][:, loc2]
        lapmmd_perm[perm_id] = torch.sum(lapker_mat11)/n1/(n1-1) + torch.sum(lapker_mat22)/n2/(n2-1) - 2 * torch.mean(lapker_mat12)


    E1_thresh = torch.quantile(ED_L1_perm, 1-alpha)
    E2_thresh = torch.quantile(ED_L2_perm, 1-alpha)
    mmd_thresh = torch.quantile(mmd_perm, 1-alpha)
    bg_thresh = torch.quantile(bg_perm, 1-alpha)
    lapmmd_thresh = torch.quantile(lapmmd_perm, 1-alpha)

    mmd_decision = (mmd>mmd_thresh)        
    edl1_decision = (edl1>E1_thresh)
    edl2_decision = (edl2>E2_thresh)
    bg_decision = (bg>bg_thresh) 
    lapmmd_decision = (lapmmd>lapmmd_thresh) 

    mmd_pval = torch.mean((mmd_perm > mmd).float())
    edl1_pval = torch.mean((ED_L1_perm > edl1).float())
    edl2_pval = torch.mean((ED_L2_perm > edl2).float())
    bg_pval = torch.mean((bg_perm > bg).float())  
    lapmmd_pval = torch.mean((lapmmd_perm > lapmmd).float())

    return {'mmd_decision': mmd_decision, 'edl1_decision': edl1_decision, 'edl2_decision': edl2_decision, 'bg_decision': bg_decision, \
             'mmd_pval': mmd_pval, 'edl1_pval': edl1_pval, 'edl2_pval': edl2_pval, 'bg_pval': bg_pval, \
             'lapmmd_decision': lapmmd_decision, 'lapmmd_pval': lapmmd_pval}