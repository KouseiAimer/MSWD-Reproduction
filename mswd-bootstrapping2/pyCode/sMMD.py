"""
    function files to calculate the studentized MMD (Gao & Shao)
"""
import torch
from sklearn.gaussian_process.kernels import RBF
from torch.distributions.normal import Normal
from pyCode.global_var import device
from pyCode.MMD import kernelSigma

def get_asymVar(ker_mat_1, ker_mat_2, ker_mat_12):
    n1 = ker_mat_1.size(0)
    n2 = ker_mat_2.size(0)
    ker_mat1 = torch.cat((ker_mat_1,ker_mat_12), 1)
    ker_mat2 = torch.cat((ker_mat_12.T, ker_mat_2), 1)
    ker_mat = torch.cat((ker_mat1, ker_mat2), 0) 
    a_col = torch.sum(ker_mat, 0) / (n1+n2-2) # row summation
    a_row = torch.sum(ker_mat, 1) / (n1+n2-2)
    a_row = a_row.reshape((n1+n2, 1))
    a_mean = torch.sum(ker_mat) / (n1+n2-1) / (n1+n2-2)
    a0 = 1
    a_col_rep = a_col.repeat((n1+n2, 1))
    a_row_rep = a_row.repeat((1, n1+n2))
    a_mean_rep = a_mean.repeat((n1+n2, n1+n2))
    A = ker_mat - a_col_rep - a_row_rep + a_mean_rep
    A = A - torch.diag(torch.diag(A))
    asymVar = torch.sum(A**2) / (n1+n2) / (n1+n2-3) - (a0**2)/(n1+n2-1)/(n1+n2-3)
    return asymVar


def get_smmd(data1, data2):
    n1 = data1.size(0)
    n2 = data2.size(0)
    sigma = kernelSigma(data1, data2)
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
    asymVar = get_asymVar(ker_mat_1, ker_mat_2, ker_mat_12)
    ker_mat_1 = ker_mat_1 - torch.diag(torch.diag(ker_mat_1))
    ker_mat_2 = ker_mat_2 - torch.diag(torch.diag(ker_mat_2))
    mmd = torch.sum(ker_mat_1)/n1/(n1-1) + torch.sum(ker_mat_2)/n2/(n2-1) - 2 * torch.mean(ker_mat_12)
    c_nm = 2/n1/(n1-1) + 4/n1/n2 + 2/n2/(n2-1)
    smmd = mmd / (c_nm * asymVar) ** 0.5
    smmd = smmd.to('cpu')
    return smmd

def sMMD_test(data1, data2, alpha=0.05):
    smmd = get_smmd(data1, data2)
    gauss = Normal(0,1)
    pval = 1-gauss.cdf(smmd)
    decision = (pval < alpha)
    return {'smmd_decision': decision, 'smmd_pval': pval}
