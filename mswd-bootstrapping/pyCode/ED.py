
import torch
from pyCode.global_var import device

def ED(data1, data2, p=2):
    """
        compute the energy distance under p-norm
    """
    data1 = data1.to(device)
    data2 = data2.to(device)
    n1 = data1.size(0)
    n2 = data2.size(0)
    dist1 = torch.cdist(data1, data1, p=p)
    dist2 = torch.cdist(data2, data2, p=p)
    dist12 = torch.cdist(data1, data2, p=p)
    ED = 2*torch.mean(dist12) - torch.sum(dist1)/n1/(n1-1) - torch.sum(dist2)/n2/(n2-1)
    temp1 = torch.cat((dist1, dist12), 1)
    temp2 = torch.cat((dist12.T, dist2), 1)
    dist_mat = torch.cat((temp1, temp2), 0)
    ED = ED.to('cpu')
    return ED, dist_mat


