""" Usage example: python main_diff_signals.py --model mean-decay --sample_dim 500 --sig sig5 --signal 0.8 """
"""
    implement the simulations under different signals
    mswd_l1: the proposed bootstrapping method with L1 sparsity
    pwdPerm: projected Wasserstein distance with k=3, permutation
"""
import torch
import numpy as np
import argparse
from pyCode.dataGenerate import dataGenerate
from pyCode.sim import sim
from pyCode.maxSlicedWD_L0_L1Approx import maxSlicedWDL0_L1Approx, maxSlicedWDL0_L1Approx_bootstrap, tune_l0
from pyCode.global_var import device

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='mean-decay', type=str, help='')
        parser.add_argument('--nrun', default=500, type=int, help='num of runs')
        parser.add_argument('--n1', default=250, type=int, help='')
        parser.add_argument('--n2', default=250, type=int, help='')
        parser.add_argument('--sample_dim', default=500, type=int, help='')
        parser.add_argument('--alpha', default=0.05, type=float, help='')
        parser.add_argument('--mswd_l1', default=1, type=int, help='')
        parser.add_argument('--perm', default=1, type=int, help='')
        parser.add_argument('--smmd', default=1, type=int, help='')
        parser.add_argument('--pwdPerm', default=1, type=int, help='')
        parser.add_argument('--wd_NNapprox', default=-1, type=int, help='')
        parser.add_argument("--signal", type=float, help="signal", default=[1.0], nargs='+')
        parser.add_argument('--sig', type=str, default='', help='')

        self.args = parser.parse_args()
        self.process_args()
    
    def process_args(self):
        self.args.signal = torch.tensor(self.args.signal).float()
        self.args.n = self.args.n1 + self.args.n2
        self.args.methods = {'mswd_l1': self.args.mswd_l1>0, 'perm': self.args.perm>0, 'smmd': self.args.smmd>0, \
                            'pwdPerm': self.args.pwdPerm>0, 'wd_NNapprox': self.args.wd_NNapprox>0}

if __name__ == '__main__':

    opts = Options()
    opts = opts.args
    print(opts.methods)

    opts.nB = 500
    n1, n2, sample_dim, nrun = opts.n1, opts.n2, opts.sample_dim, opts.nrun
    sig = opts.sig
    signal = opts.signal
    model = opts.model
    opts.tune = True
    opts.lams = torch.exp(torch.linspace(np.log(1.5), np.log(5), steps=5))
    opts.reps = 10
    s=None
    lam_l0_seq = torch.tensor([1, 5, 10, 20, 50])

    hyper_params = {'learn_rate': 0.001, 'nepoch': 1000, \
        'in_features': sample_dim, 'hidden_layers': 3, 'node': 200}

    opts.hyper_params = hyper_params

    if model == 'joint':
        if sig == 'sig5':
            signal = torch.ones(100)
        elif sig == 'sig4':
            signal = torch.ones(80)
        elif sig == 'sig3':
            signal = torch.ones(60)
        elif sig == 'sig2':
            signal = torch.ones(40)
        else:
            signal = torch.ones(20)

    mmd_decision = np.zeros(nrun)
    edl1_decision = np.zeros(nrun)
    edl2_decision = np.zeros(nrun)
    bg_decision = np.zeros(nrun)
    mswd_l1_decision = np.zeros(nrun)
    mswd_l0_decision = np.zeros(nrun)
    kPWD_perm_decision = np.zeros(nrun)
    smmd_decision = np.zeros(nrun)
    wd_nnapprox_decision = np.zeros(nrun)
    for mcrun in range(nrun):
        print('run {}'.format(mcrun))
        data1, data2 = dataGenerate(n1, n2, sample_dim, signal, model, s=s)
        data1 = data1.to(device)
        data2 = data2.to(device)
        p1 = torch.ones(n1, device=device)/n1
        p2 = torch.ones(n2, device=device)/n2
        scale = (n1*n2/(n1+n2))**0.5
        
        # the proposed test with l0 sparsity
        lam_l0 = tune_l0(data1, data2, lam_l0_seq, k=2, reps=5)
        lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam_l0**0.5), steps=10))  
        mswd_l0, _, _ = maxSlicedWDL0_L1Approx(data1, data2, p1, p2, lam_l0, lam_l1, candidate_adaptive=True, n_l1=10, reps=10)
        Tn = scale * mswd_l0
        mswd_l0_thresh, mswd_l0_boots_sample = maxSlicedWDL0_L1Approx_bootstrap(data1, data2, lam_l0, lam_l1, candidate_adaptive=True, n_l1=10, reps=1, B=300)
        mswd_l0_decision[mcrun] = (Tn>mswd_l0_thresh)
        mswd_l0_rejPerc = np.mean(mswd_l0_decision[0:(mcrun + 1)])
        print('run {}, percentage of rejection: max-sliced wd with l0 sparsity {}'.format(mcrun, mswd_l0_rejPerc))
        
        # other methods
        result = sim(data1, data2, opts)
        if opts.methods['perm'] == True:
            mmd_decision[mcrun] = result['perm']['mmd_decision']
            edl1_decision[mcrun] = result['perm']['edl1_decision']
            edl2_decision[mcrun] = result['perm']['edl2_decision']
            bg_decision[mcrun] = result['perm']['bg_decision']

            mmd_rejPerc = np.mean(mmd_decision[0:(mcrun+1)])
            edl1_rejPerc = np.mean(edl1_decision[0:(mcrun + 1)])
            edl2_rejPerc = np.mean(edl2_decision[0:(mcrun + 1)])
            bg_rejPerc = np.mean(bg_decision[0:(mcrun + 1)])

            print('run {}, percentage of rejection: mmd {}\n, edl1 {}, edl2 {}, bg {} \
                '.format(mcrun, mmd_rejPerc, edl1_rejPerc, edl2_rejPerc, bg_rejPerc))

        if opts.methods['mswd_l1'] == True:
            mswd_l1_decision[mcrun] = result['mswd_l1_decision']
            mswd_l1_rejPerc = np.mean(mswd_l1_decision[0:(mcrun + 1)])

            print('run {}, percentage of rejection: max-sliced wd with l1 sparsity {}'.format(mcrun, mswd_l1_rejPerc))

        if opts.methods['smmd'] == True:
            smmd_decision[mcrun] = result['smmd_decision']
            smmd_rejPerc = np.mean(smmd_decision[0:(mcrun + 1)])
            print('run {}, percentage of rejection: smmd {}'.format(mcrun, smmd_rejPerc))

        if opts.methods['pwdPerm'] == True:
            kPWD_perm_decision[mcrun] = result['kPWD_perm_decision']
            pwd_perm_rejPerc = np.mean(kPWD_perm_decision[0:(mcrun+1)])
            print('run {}, percentage of rejection: pwd perm k3 {}'.format(mcrun, pwd_perm_rejPerc))           

        if opts.methods['wd_NNapprox'] == True:
            wd_nnapprox_decision[mcrun] = result['wd_nnapprox_decision']
            wd_nnapprox_rejPerc = np.mean(wd_nnapprox_decision[0:(mcrun + 1)])
            print('run {}, percentage of rejection: wd nnapprox {}'.format(mcrun, wd_nnapprox_rejPerc))


