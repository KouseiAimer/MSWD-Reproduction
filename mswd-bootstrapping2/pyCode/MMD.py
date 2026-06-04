
"""
    functions to calculate the bandwidth in MMD (median pairwise distance)
"""
import torch
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from pyCode.global_var import device

def kernelSigma(data1, data2):
    """
        compute the parameter in kernels: median distance
    """
    data1 = data1.to(device)
    data2 = data2.to(device)
    data = torch.cat((data1, data2), 0)
    square_distance = torch.cdist(data, data, p=2)**2
    square_distance = square_distance - torch.triu(square_distance)
    # rbf() has factor two in kernel
    sigma = torch.sqrt(0.5 * torch.median(square_distance[square_distance > 0])).to('cpu').numpy()  # median distance of aggregate sample
    return sigma

# def mmdSpec(data1, data2, sigma, alpha=0.05, nsample_mmdspec=300):
#     """
#         MMD spectrum consistent test for equal sample size n1=n2=m
#         V-statistic: biased mmd (m*MMD_b^2)
#     """
#     data = torch.cat((data1, data2), 0).to('cpu')
#     n = data.size(0)   
#     rbf = RBF(sigma/2**0.5)
#     ker_mat = torch.from_numpy(rbf(data)).float().to(device) #input of rbf must be on cpu, its output is numpy array
#     H = (torch.eye(n) - torch.ones((n, n)) / n).to(device)
#     gram_mat = torch.matmul(torch.matmul(H, ker_mat), H)
#     evals, _ = torch.eig(gram_mat)
#     evals, _ = torch.sort(evals[:, 0], descending=True)
#     num_evals = n - 2
#     evals = torch.abs(evals[0:num_evals]) / n
#     evals = evals.to('cpu').numpy()
#     mmdspec_sample = np.zeros(nsample_mmdspec)
#     for spec_id in range(nsample_mmdspec):
#         mmdspec_sample[spec_id] = 2 * np.sum(evals * np.random.normal(0, 1, evals.shape[0]) ** 2)
#     mmdspe_thresh = np.quantile(mmdspec_sample, q=1-alpha)
#     return mmdspe_thresh

def get_mmd(data1, data2):
    """ compute biased mmd """
    sigma = kernelSigma(data1, data2)
    # rbf() has factor two in kernel
    # rbf = RBF(sigma/2**0.5)
    # ker_mat_1 = rbf(data1) # inputs are numpy array
    # ker_mat_2 = rbf(data2)
    # ker_mat_12 = rbf(data1, data2)
    # mmd = np.mean(ker_mat_1) + np.mean(ker_mat_2) - 2 * np.mean(ker_mat_12)
    ker_mat_1 = torch.exp(-torch.cdist(data1, data1)**2/sigma**2)
    ker_mat_2 = torch.exp(-torch.cdist(data2, data2)**2/sigma**2)
    ker_mat_12 = torch.exp(-torch.cdist(data1, data2)**2/sigma**2)
    mmd = torch.mean(ker_mat_1) + torch.mean(ker_mat_2) - 2*torch.mean(ker_mat_12)
    mmd = mmd.to('cpu')
    return mmd     

def get_mmd_unbiased(data1, data2):
    """ compute unbiased mmd """
    n1 = data1.size(0)
    n2 = data2.size(0)
    sigma = kernelSigma(data1, data2)
    # rbf = RBF(sigma/2**0.5)
    # ker_mat_1 = rbf(data1)
    # ker_mat_2 = rbf(data2)
    # ker_mat_12 = rbf(data1, data2)
    # ker_mat_1 = ker_mat_1 - np.diag(np.diag(ker_mat_1))
    # ker_mat_2 = ker_mat_2 - np.diag(np.diag(ker_mat_2))
    # mmd_unbiased = np.sum(ker_mat_1)/n1/(n1-1) + np.sum(ker_mat_2)/n2/(n2-1) - 2 * np.mean(ker_mat_12)
    ker_mat_1 = torch.exp(-torch.cdist(data1, data1)**2/sigma**2)
    ker_mat_2 = torch.exp(-torch.cdist(data2, data2)**2/sigma**2)
    ker_mat_12 = torch.exp(-torch.cdist(data1, data2)**2/sigma**2)
    ker_mat_1 = ker_mat_1 - torch.diag(torch.diag(ker_mat_1))
    ker_mat_2 = ker_mat_2 - torch.diag(torch.diag(ker_mat_2))
    mmd_unbiased = torch.sum(ker_mat_1)/n1/(n1-1) + torch.sum(ker_mat_2)/n2/(n2-1) - 2 * torch.mean(ker_mat_12)
    mmd_unbiased = mmd_unbiased.to('cpu')
    return mmd_unbiased




