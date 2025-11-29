import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch

def calcMI(z1, z2):
    if isinstance(z1, torch.Tensor):
        z1 = z1.cpu().numpy()  
    if isinstance(z2, torch.Tensor):
        z2 = z2.cpu().numpy()  
    
    eps = 10e-16
    P = z1 @ z2.T  
    PXY = P / P.sum()  
    PXPY = np.outer(PXY.sum(1, keepdims=True), PXY.sum(0, keepdims=True))
    MI = np.sum(PXY * np.log(eps + PXY / (eps + PXPY)))
    return MI

def calcNMI(z1, z2):
    mi = calcMI(z1, z2)
    return (2 * mi) / (calcMI(z1, z1) + calcMI(z2, z2))  


def ArchetypeConsistency(XC1, XC2, mSST):

    if isinstance(XC1, torch.Tensor):
        XC1 = XC1.cpu().numpy()  
    if isinstance(XC2, torch.Tensor):
        XC2 = XC2.cpu().numpy()  
    
    combined = np.hstack((XC1, XC2)).T  
    D = squareform(pdist(combined, 'euclidean'))**2  
    D = D[:XC1.shape[1], XC1.shape[1]:]  
    
    i, j, v = [], [], []
    K = XC1.shape[1]
    for _ in range(K):
        min_idx = np.unravel_index(np.argmin(D), D.shape)
        i.append(min_idx[0])
        j.append(min_idx[1])
        v.append(D[min_idx])
        D[min_idx[0], :] = np.inf  
        D[:, min_idx[1]] = np.inf
    
    consistency = 1 - np.mean(v) / mSST
    
    D2 = np.abs(np.corrcoef(combined))  
    D2 = D2[:K, K:]  
    ISI = (np.sum(D2 / np.max(D2, axis=1, keepdims=True) + 
                  D2 / np.max(D2, axis=0, keepdims=True)) - 2 * K) / (2 * K * (K - 1))
    
    return consistency, ISI
