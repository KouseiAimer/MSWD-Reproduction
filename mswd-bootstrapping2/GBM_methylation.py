# implement the data analysis
import torch
import numpy as np
import math
from scipy.stats import wasserstein_distance
from pyCode.maxSlicedWD_L0_L1Approx import maxSlicedWDL0_L1Approx, maxSlicedWDL0_L1Approx_bootstrap, tune_l0
from pyCode import myProj
from pyCode.global_var import device

alpha = 0.05
data1 = np.loadtxt('GBM/GBM_prognostic_low.txt')
data2 = np.loadtxt('GBM/GBM_prognostic_high.txt')
n1, sample_dim = data1.shape
n2 = data2.shape[0]
n = n1 + n2

nB = 500
lam_l0_seq = torch.exp(torch.linspace(math.log(1), math.log(50), steps=20))
reps = 10
data1 = torch.from_numpy(data1).float().to(device)
data2 = torch.from_numpy(data2).float().to(device)
p1 = torch.ones(n1)/n1
p2 = torch.ones(n2)/n2
# tune the sparsity parameter with CV
lam_l0 = tune_l0(data1, data2, lam_l0_seq, k=2, reps=10)
lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam_l0**0.5), steps=10))  
mswd_l0, V_opt, _ = maxSlicedWDL0_L1Approx(data1, data2, p1, p2, lam_l0, lam_l1, candidate_adaptive=True, reps=10)
scale = (n1*n2/(n1+n2))**0.5
statistic = scale * mswd_l0
V_opt = V_opt.to('cpu')
mswd_thresh, mswd_boots_sample = maxSlicedWDL0_L1Approx_bootstrap(data1, data2, lam_l0, lam_l1, candidate_adaptive=True, reps=1, B=500)
mswd_pval = torch.mean((mswd_boots_sample > statistic).float())


# investigate marginal difference
data1 = data1.to('cpu')
data2 = data2.to('cpu')
stat_rec = torch.zeros(sample_dim)
is_significant = torch.zeros(sample_dim)
pvals = torch.zeros(sample_dim)
for j in range(sample_dim):
    stat_rec[j] = scale * wasserstein_distance(data1[:,j], data2[:,j], p1, p2)
    pvals[j] = torch.mean((mswd_bs>stat_rec[j]).float())
    is_significant[j] = stat_rec[j] > (torch.quantile(mswd_bs, 1-alpha))
# np.savetxt('GBM/GBM_grade_marginal_pvals.txt', pvals)

# genes in both prognostic genes and some GO terms (BP)
gene_sets = ['mitotic_cell_cycle', 'mitotic_cell_cycle_process', 'cell_cycle_process', \
            'cell_cycle', 'DNA_metabolic_process', 'cell_division', 'mitotic_nuclear_division']
data1 = data1.to(device)
data2 = data2.to(device)
statistics = torch.zeros(len(gene_sets))
for l in range(len(gene_sets)):
    gene_set_name = gene_sets[l]
    idx = np.loadtxt('GBM/idx_genes_{}.txt'.format(gene_set_name))
    idx = idx - 1
    idx = idx.astype(int)
    mswd_l0, V_opt, _ = maxSlicedWDL0_L1Approx(data1[:,idx], data2[:,idx], p1, p2, lam_l0, lam_l1, candidate_adaptive=True, reps=10)     
    statistics[l] = scale * mswd_l0
    V_opt = V_opt.to('cpu')
    # np.savetxt('GBM/direction_{}.txt'.format(gene_set_name), V_opt)
