
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import numpy as np
from mswd_bootstrapping import mswdtest, mswd_sci_direction, mswd_sci_marginal

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

n1 = 250
n2 = 250
sample_dim = 500
rand_generator = MultivariateNormal(torch.zeros(sample_dim, device=device), torch.eye(sample_dim, device=device))
data1 = rand_generator.sample([n1])
signal = torch.ones(1,device=device)*0.8
mean =  signal / torch.arange(1,sample_dim+1,device=device)**3
rand_generator = MultivariateNormal(mean, torch.eye(sample_dim, device=device))
data2 = rand_generator.sample([n2])
lam = torch.tensor([1, 5, 10, 20, 50]) # candidate set of sparsity parameters

out = mswdtest(data1, data2, lam) # two-sample test
print('the decision of the test: {}'.format(out['reject'])) # 1: reject the null hypothesis; 0: not reject the null hypothesis
V_opt = out['optimal direction']
pval = out['pvalue']
lam_selected = out['selected lam']
bsample = out['bootstrap sample']

V = torch.zeros(sample_dim, 2)
V[0,0] = 1 # directions of interest
V[1,1] = 1
out_sci = mswd_sci_direction(data1, data2, V, lam_selected, bsample)
print(out_sci['sci bounds'])

idx = [[0], [1,2]] # index sets of marginal coordinates of interest
out_sci = mswd_sci_marginal(data1, data2, idx, lam_selected, bsample)
print(out_sci['sci bounds'])
