"""
    use NN to approximate the Lipschitz function class
    bootstrapping the Wasserstein distance
    Neural computation (2022). "Hypothesis Test and Confidence Analysis With Wasserstein Distance on General Dimension"
"""

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
from pyCode.global_var import device

class wdnet(nn.Module):
    def __init__(self, hyper_params):
        super().__init__() 
        self.in_features = hyper_params['in_features']
        self.node = hyper_params['node']
        self.hidden_layers = hyper_params['hidden_layers']
        net_list = []
        net_list.append(nn.utils.spectral_norm(nn.Linear(self.in_features, self.node)))
        for i in range(self.hidden_layers-1):
            net_list.append(nn.utils.spectral_norm(nn.Linear(self.node, self.node)))
        net_list.append(nn.utils.spectral_norm(nn.Linear(self.node, 1)))
        
        self.layers = nn.ModuleList(net_list)
        # self.relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layers[0](x))
        for i in range(self.hidden_layers-1):
            x = self.relu(self.layers[i+1](x))
        x = self.layers[-1](x)
        return x

def get_WDapprox(data1, data2, hyper_params, show_plot=False):

    learn_rate = hyper_params['learn_rate']
    nepoch = hyper_params['nepoch']
    n1 = data1.size(0)
    n2 = data2.size(0)
    sample1 = data1.to(device)
    sample2 = data2.to(device)

    mywdnet = wdnet(hyper_params)
    mywdnet.to(device)
    optimizer = optim.Adam(mywdnet.parameters(), lr=learn_rate, betas=(0.5, 0.999))
    # optimizer = optim.SGD(mywdnet.parameters(), lr=learn_rate, momentum=0.9)
    error_rec = []
    mywdnet.train()
    for epoch in range(nepoch):
        # update net parameters
        out1 = mywdnet(sample1)
        out2 = mywdnet(sample2)
        error = out1.mean() - out2.mean()
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        error_rec.append(error.detach().item())
        if show_plot == True:
            if (epoch+1) % 100 == 0:
                plt.clf()
                plt.plot(error_rec)
                plt.savefig('wdTrainerror.png')
    mywdnet.eval()
    with torch.no_grad():
        wasser_distance = mywdnet(data2).mean() - mywdnet(data1).mean()
        rho = n1*n2/(n1+n2)
        Tn = np.sqrt(rho) * wasser_distance
        Tn = Tn.to('cpu')
        print('approximated WD test statistic {}'.format(Tn))
    return Tn

def WDapprox_bootstrap(data, hyper_params, r, B=500, alpha=0.05, show_plot=False):
    """
        r: percentages
        output the quantiles at these percentages
    """
    n = data.size(0)   
    learn_rate = hyper_params['learn_rate']
    nepoch = hyper_params['nepoch']
    sample = data.to(device)
    Z = torch.zeros(B)
    for i in range(B):
        # gauss = torch.randn(n, 1, device=device)
        gauss = np.random.normal(0,1,(n,1))
        gauss = torch.from_numpy(gauss).float().to(device)
        mywdnet = wdnet(hyper_params)
        mywdnet.to(device)
        optimizer = optim.Adam(mywdnet.parameters(), lr=learn_rate, betas=(0.5, 0.999))
        # optimizer = optim.SGD(mywdnet.parameters(), lr=learn_rate, momentum=0.9)
        error_rec = []
        mywdnet.train()
        for epoch in range(nepoch):
            # update net parameters
            out = mywdnet(sample)
            error = -(gauss * (out - out.mean())).mean() 
            optimizer.zero_grad()
            error.backward()
            optimizer.step()          
            error_rec.append(error.detach().item() )
            if show_plot == True:
                if (epoch+1) % 100 == 0:
                    plt.clf()
                    plt.plot(error_rec)
                    plt.savefig('bootstrapwdTrainerror.png')
        mywdnet.eval()
        with torch.no_grad():
            Z[i] = np.sqrt(n) * (gauss * (mywdnet(data) - mywdnet(data).mean())).mean()
            if (i+1)%100 == 0:
                print('WD multiplier bootstrap {}, current quantile of z: {}'.format(i+1, torch.quantile(Z[0:(i+1)], 1-alpha)))
    pers = 1 - r*alpha
    quantiles = torch.quantile(Z, pers)
    quantiles = quantiles.to('cpu')
    return quantiles
