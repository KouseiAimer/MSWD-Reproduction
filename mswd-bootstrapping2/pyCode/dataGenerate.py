import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.poisson import Poisson
from pyCode.global_var import device

def correlation(sample_dim, band, rho=0.5):
    R = 0
    for i in range(band):
        if i==0:
            R = R + torch.diag(torch.ones(sample_dim-i, device=device) * rho**abs(i), i)
        else:
            R = R + torch.diag(torch.ones(sample_dim-i, device=device) * rho**abs(i), i) + torch.diag(torch.ones(sample_dim-i, device=device) * rho**abs(i), -i) 
    return R

def dataGenerate(n1, n2, sample_dim, signal, model, s=None):
    """
        signal: tensor
        model: distributional differences
    """
    signal = signal.to(device)
    signal_len = signal.size(0)

    if model=='null-gauss':
        # Gaussian
        R = correlation(sample_dim, band=sample_dim)
        rand_generator = MultivariateNormal(torch.zeros(sample_dim, device=device), R)
        data1 = rand_generator.sample([n1])
        data2 = rand_generator.sample([n2])
    if model=='null-mix_gauss':
        # mixture Gaussian
        labels = torch.bernoulli(0.5*torch.ones(n1,1, device=device))
        gen1 = MultivariateNormal(-1*torch.ones(sample_dim, device=device), torch.eye(sample_dim, device=device))
        gen2 = MultivariateNormal(torch.ones(sample_dim, device=device), torch.eye(sample_dim, device=device))
        data1 = gen1.sample([n1])*labels + (1-labels) * gen2.sample([n1])
        labels = torch.bernoulli(0.5*torch.ones(n2,1, device=device))
        data2 = gen1.sample([n2])*labels + (1-labels) * gen2.sample([n2])     
    if model=='mean':
        R = correlation(sample_dim, band=sample_dim) # bandable correlation
        rand_generator = MultivariateNormal(torch.zeros(sample_dim, device=device), R)
        data1 = rand_generator.sample([n1])
        mean = torch.zeros(sample_dim, device=device)
        if s==None:
            s=sample_dim
        mean[range(s)] = torch.ones(s, device=device) * signal / (s**0.5)
        rand_generator = MultivariateNormal(mean, R)
        data2 = rand_generator.sample([n2])     
    if model=='mean-decay':
        R = correlation(sample_dim, band=sample_dim) 
        rand_generator = MultivariateNormal(torch.zeros(sample_dim, device=device), R)
        data1 = rand_generator.sample([n1])
        mean = signal / torch.arange(1,sample_dim+1,device=device)**3
        rand_generator = MultivariateNormal(mean, R)
        data2 = rand_generator.sample([n2])     
    if model=='var': # different variances, same off-diagonal elements of covariance 
        R = correlation(sample_dim, band=sample_dim)
        gen = MultivariateNormal(torch.zeros(sample_dim, device=device), R)
        data1 = gen.sample([n1])
        cov = R.clone()
        if s==None:
            s = sample_dim
        cov[range(s), range(s)] = torch.ones(s, device=device)*signal/(s**0.5)+1
        gen = MultivariateNormal(torch.zeros(sample_dim, device=device), cov)
        data2 = gen.sample([n2])      
    if model=='var-decay': # different variances, same off-diagonal elements of covariance 
        R = correlation(sample_dim, band=sample_dim) # when band=1, it becomes an identity matrix
        cov = R.clone()
        gen = MultivariateNormal(torch.zeros(sample_dim, device=device), cov)
        data1 = gen.sample([n1])
        cov = R.clone()
        cov[range(sample_dim), range(sample_dim)] = signal/torch.arange(1, sample_dim+1, device=device)**3+1 # replace diagonal elements
        gen = MultivariateNormal(torch.zeros(sample_dim, device=device), cov)
        data2 = gen.sample([n2])   
    if model == 'var-trace':
        cov = correlation(sample_dim, band=sample_dim)
        s_tmp = 2*(s//2)
        cov[range(s_tmp), range(s_tmp)] = torch.cat((signal/(s**0.5)*torch.ones(s//2, device=device)+1, torch.ones(s//2, device=device))) 
        gen = MultivariateNormal(torch.zeros(sample_dim, device=device), cov)
        data1 = gen.sample([n1])
        cov[range(s_tmp), range(s_tmp)] = torch.cat((torch.ones(s//2, device=device), 1+signal/(s**0.5)*torch.ones(s//2, device=device))) 
        gen = MultivariateNormal(torch.zeros(sample_dim, device=device), cov)
        data2 = gen.sample([n2])
    if model=='marginal':
        labels = torch.bernoulli(0.5*torch.ones(n1,1, device=device))
        gen1 = MultivariateNormal(-1*torch.ones(sample_dim, device=device), torch.eye(sample_dim, device=device))
        gen2 = MultivariateNormal(torch.ones(sample_dim, device=device), torch.eye(sample_dim, device=device))
        data1 = gen1.sample([n1])*labels + (1-labels) * gen2.sample([n1])
        R = correlation(signal_len, band=signal_len)
        V = torch.diag(torch.ones(signal_len, device=device)*2**0.5)
        cov = torch.matmul(torch.matmul(V, R), V)
        gen = MultivariateNormal(torch.zeros(signal_len, device=device), cov)
        temp1 = gen.sample([n2])
        if (sample_dim-signal_len) > 0:
            labels = torch.bernoulli(0.5*torch.ones(n2,1, device=device))
            gen1 = MultivariateNormal(-1*torch.ones(sample_dim-signal_len, device=device), torch.eye(sample_dim-signal_len, device=device))
            gen2 = MultivariateNormal(torch.ones(sample_dim-signal_len, device=device), torch.eye(sample_dim-signal_len, device=device))
            temp2 = gen1.sample([n2])*labels + (1-labels) * gen2.sample([n2])
            data2 = torch.cat((temp1, temp2), 1)
        else:
            data2 = temp1        
    if model=='joint':
        labels = torch.bernoulli(0.5*torch.ones(n1,1, device=device))
        gen1 = MultivariateNormal(-1*torch.ones(sample_dim, device=device), torch.eye(sample_dim, device=device))
        gen2 = MultivariateNormal(torch.ones(sample_dim, device=device), torch.eye(sample_dim, device=device))
        data1 = gen1.sample([n1])*labels + (1-labels) * gen2.sample([n1])
        R = correlation(signal_len, band=signal_len, rho=0.9)
        covmat = torch.eye(sample_dim, device=device)
        covmat[0:signal_len, 0:signal_len] = R # replace upper block matrix
        gen1 = MultivariateNormal(-1*torch.ones(sample_dim, device=device), covmat)
        gen2 = MultivariateNormal(torch.ones(sample_dim, device=device), covmat)
        labels = torch.bernoulli(0.5*torch.ones(n2,1, device=device))
        data2 = gen1.sample([n2])*labels + (1-labels) * gen2.sample([n2])
    if model=='high-order-normal-pois':
        gen = MultivariateNormal(torch.ones(sample_dim, device=device), torch.eye(sample_dim, device=device))
        data1 = gen.sample([n1])
        gen = Poisson(torch.ones(1, device=device))
        temp1 = gen.sample([n2*signal_len]).reshape(n2, signal_len)
        if (sample_dim - signal_len)>0:
            gen = MultivariateNormal(torch.ones(sample_dim-signal_len, device=device), torch.eye(sample_dim-signal_len, device=device))
            temp2 = gen.sample([n2])
            data2 = torch.cat((temp1, temp2), 1)
        else:
            data2 = temp1 

    return data1, data2



