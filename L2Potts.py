import torch
import numpy as np


class L2Potts():
    def __init__(self, data, weights, gamma, device=None) -> None:
        self.mData =  data #1x 3 x n 3 channel n pixels
        self.mWeights = weights
        self.mGamma = gamma
        self.mExludedIntervalSize = 0
        self.device = device if device else 'cpu'
        self.mExcludedIntervalSize = 0
    def setExcludedIntervalSize(self, excludedIntervalSize):
        self.mExcludedIntervalSize = excludedIntervalSize

    def getWeight(self, i):
        if self.mWeights is None:
            return 1
        else:
            return self.mWeights[i]

    # mData (1,3,n) n pixels 3 channel

    def __call__(self):
        # get the data shape 1 x 3 x n (3 channels, n pixels)
        _, nVec, n = self.mData.shape
        arrJ = torch.zeros((n,), device=self.device)
        
        arrP = torch.zeros((n,), device=self.device)
        d = 0
        p = 0
        dpg = 0

        # m will be a list of torch tensors. 
        # !!Maybe implement as just torch tensor with correct dimension
        m = torch.zeros((nVec, n+1), device=self.device)

        s = torch.zeros((n+1,), device=self.device)
        w = torch.zeros((n+1,), device=self.device)

        s[0] = 0
        wTemp = None
        mTemp = None 
        wDiffTemp = None

        for j in range(n):
            wTemp = self.getWeight(j)
            m[:,j+1] = torch.multiply(self.mData[0,:,j],wTemp)
            m[:,j+1] = m[:,j+1] + m[:,j]
            s[j+1] = torch.sum(torch.pow(self.mData[0,:,j],2))*wTemp + s[j]
            w[j+1] = w[j] + wTemp


        for r in range(1, n+1):
            arrP[r-1] = s[r] - self._normQuad(m[:,r], 0) / (w[r])
            arrJ[r-1] = 0
            for l in range(r- self.mExcludedIntervalSize,1,-1):
                mTemp = torch.sum(torch.pow(m[:,r] - m[:,l-1],2),0)
            
                wDiffTemp = w[r] - w[l-1]
                if wDiffTemp == 0:
                    d = 0
                else:
                    d = s[r] - s[l-1] - mTemp / wDiffTemp

                dpg = d + self.mGamma

                if dpg > arrP[r-1]:
                    break
                p = arrP[l-2] + dpg
                if p < arrP[r-1]:
                    arrP[r-1] = p
                    arrJ[r-1] = l-1
        
        r = n
        l = int(arrJ[r-1])
        mu = torch.zeros((nVec,), device=self.device)

        while r > 0:
            mu = torch.divide(m[:,r] - m[:,l], (w[r] - w[l]))
            
            for k in range(nVec):
                self.mData[:,k,1:r] = mu[k]

            r = l
            if r < 1:
                break
            l = int(arrJ[r-1])

        return self.mData

    def _normQuad(self, x, sum_dim=1):
        return torch.sum(torch.pow(x,2),dim=sum_dim)
            


