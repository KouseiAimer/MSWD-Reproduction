# implement the data analysis
import torch
import numpy as np
import math
from pathlib import Path
from scipy.stats import wasserstein_distance
from pyCode.maxSlicedWD_L0_L1Approx import maxSlicedWDL0_L1Approx, maxSlicedWDL0_L1Approx_bootstrap, tune_l0
from pyCode.global_var import device

out_dir = Path('GBM')
out_dir.mkdir(exist_ok=True)
alpha = 0.05
data1 = np.loadtxt(out_dir / 'GBM_prognostic_low.txt')
data2 = np.loadtxt(out_dir / 'GBM_prognostic_high.txt')
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
lam_l0 = tune_l0(data1, data2, lam_l0_seq, k=2, reps=reps)
lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam_l0**0.5), steps=10))  
mswd_l0, V_opt, _ = maxSlicedWDL0_L1Approx(data1, data2, p1, p2, lam_l0, lam_l1, candidate_adaptive=True, reps=reps)
scale = (n1*n2/(n1+n2))**0.5
statistic = scale * mswd_l0
V_opt = V_opt.to('cpu')
mswd_thresh, mswd_boots_sample = maxSlicedWDL0_L1Approx_bootstrap(data1, data2, lam_l0, lam_l1, candidate_adaptive=True, reps=1, B=nB)
mswd_pval = torch.mean((mswd_boots_sample > statistic).float())

np.savetxt(out_dir / 'GBM_global_direction.txt', V_opt.numpy())
np.savetxt(out_dir / 'GBM_global_bootstrap_sample.txt', mswd_boots_sample.to('cpu').numpy())
np.savetxt(
    out_dir / 'GBM_global_test_summary.txt',
    np.array([
        ['n1', n1],
        ['n2', n2],
        ['sample_dim', sample_dim],
        ['alpha', alpha],
        ['B', nB],
        ['selected_lam_l0', lam_l0.item()],
        ['mswd_l0', mswd_l0.item()],
        ['scale', scale],
        ['test_statistic', statistic.item()],
        ['bootstrap_threshold', mswd_thresh.item()],
        ['p_value', mswd_pval.item()],
        ['reject', int((statistic > mswd_thresh).item())],
    ], dtype=object),
    fmt='%s',
)


# investigate marginal difference
data1 = data1.to('cpu')
data2 = data2.to('cpu')
data1_np = data1.numpy()
data2_np = data2.numpy()
p1_np = p1.numpy()
p2_np = p2.numpy()
mswd_boots_sample = mswd_boots_sample.to('cpu')
mswd_thresh = mswd_thresh.to('cpu')
stat_rec = torch.zeros(sample_dim)
is_significant = torch.zeros(sample_dim)
pvals = torch.zeros(sample_dim)
for j in range(sample_dim):
    stat_rec[j] = scale * wasserstein_distance(data1_np[:,j], data2_np[:,j], p1_np, p2_np)
    pvals[j] = torch.mean((mswd_boots_sample>stat_rec[j]).float())
    is_significant[j] = stat_rec[j] > mswd_thresh
np.savetxt(out_dir / 'GBM_grade_marginal_pvals.txt', pvals.numpy())
np.savetxt(out_dir / 'GBM_grade_marginal_statistics.txt', stat_rec.numpy())
np.savetxt(out_dir / 'GBM_grade_marginal_is_significant.txt', is_significant.numpy(), fmt='%d')
marginal_summary = np.column_stack([
    np.arange(1, sample_dim + 1).astype(str),
    stat_rec.numpy().astype(str),
    pvals.numpy().astype(str),
    is_significant.numpy().astype(int).astype(str),
])
np.savetxt(
    out_dir / 'GBM_grade_marginal_summary.txt',
    marginal_summary,
    fmt='%s',
    header='gene_index_1based statistic p_value is_significant',
    comments='',
)

# genes in both prognostic genes and some GO terms (BP)
gene_sets = ['mitotic_cell_cycle', 'mitotic_cell_cycle_process', 'cell_cycle_process', \
            'cell_cycle', 'DNA_metabolic_process', 'cell_division', 'mitotic_nuclear_division']
data1 = data1.to(device)
data2 = data2.to(device)
statistics = torch.zeros(len(gene_sets))
gene_set_pvals = torch.zeros(len(gene_sets))
gene_set_is_significant = torch.zeros(len(gene_sets))
gene_set_sizes = np.zeros(len(gene_sets), dtype=int)
for l in range(len(gene_sets)):
    gene_set_name = gene_sets[l]
    idx = np.atleast_1d(np.loadtxt(out_dir / 'idx_genes_{}.txt'.format(gene_set_name))).astype(int) - 1
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
    gene_set_sizes[l] = len(idx)
    mswd_l0, V_opt, _ = maxSlicedWDL0_L1Approx(data1[:,idx_t], data2[:,idx_t], p1, p2, lam_l0, lam_l1, candidate_adaptive=True, reps=reps)
    statistics[l] = scale * mswd_l0
    gene_set_pvals[l] = torch.mean((mswd_boots_sample>statistics[l].to('cpu')).float())
    gene_set_is_significant[l] = statistics[l].to('cpu') > mswd_thresh
    V_opt = V_opt.to('cpu')
    np.savetxt(out_dir / 'direction_{}.txt'.format(gene_set_name), V_opt.numpy())

gene_set_summary = np.column_stack([
    np.array(gene_sets),
    gene_set_sizes.astype(str),
    statistics.numpy().astype(str),
    gene_set_pvals.numpy().astype(str),
    gene_set_is_significant.numpy().astype(int).astype(str),
])
np.savetxt(
    out_dir / 'GBM_gene_set_summary.txt',
    gene_set_summary,
    fmt='%s',
    header='gene_set n_genes statistic p_value is_significant',
    comments='',
)

print('Global MSWD test statistic: {}'.format(statistic.item()))
print('Global MSWD bootstrap threshold: {}'.format(mswd_thresh.item()))
print('Global MSWD p-value: {}'.format(mswd_pval.item()))
print('Results saved in {}'.format(out_dir.resolve()))
