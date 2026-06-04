
import numpy as np
import torch
from pyCode.global_var import device
from pyCode.PWD import PWD

def pwdPerm(data1, data2, alpha=0.05, n_perm=500):

    data1 = data1.to(device)
    data2 = data2.to(device)
    n1, sample_dim = data1.size()
    n2 = data2.size(0)
    p1 = torch.ones(n1)/n1
    p2 = torch.ones(n2)/n2
    # V0 = torch.randn(sample_dim, 3)
    V0 = np.random.normal(0,1,(sample_dim,3))
    V0 = torch.from_numpy(V0).float()
    V0, _ = torch.linalg.qr(V0)
    kPWD_val = PWD(data1, data2, V0, p1, p2)         
    print('pwd perm k=3 calculate the test statistic, value {}'.format(kPWD_val))
    data = torch.cat((data1, data2), 0)
    kPWD_perm =  torch.zeros(n_perm)
    for i in range(n_perm):
        # loc = np.random.choice(n1+n2, n1, replace=False)
        # loc2 = np.delete(np.arange(n1+n2), loc)
        # perm_data1 = data[loc, :]
        # perm_data2 = data[loc2, :]
        # locperm = torch.randperm(n1+n2, device=device)
        locperm = np.random.permutation(n1+n2)
        perm_data1 = data[locperm[0:n1], :]
        perm_data2 = data[locperm[n1:(n1+n2)], :]
        # V0 = torch.randn(sample_dim, 3)
        V0 = np.random.normal(0,1,(sample_dim,3))
        V0 = torch.from_numpy(V0).float()
        V0, _ = torch.linalg.qr(V0)
        kPWD_perm[i] = PWD(perm_data1, perm_data2, V0, p1, p2)
        if (i+1) % 100==0:
            print('kPWD {}-th permutation, thresh {}'.format(i+1, torch.quantile(kPWD_perm[0:(i+1)], 1-alpha)))
    kPWD_perm_decision = (kPWD_val > torch.quantile(kPWD_perm, 1-alpha))
    kPWD_perm_pval = torch.mean((kPWD_perm > kPWD_val).float())
  
    return {'kPWD_perm_decision': kPWD_perm_decision, 'kPWD_perm_pval': kPWD_perm_pval}