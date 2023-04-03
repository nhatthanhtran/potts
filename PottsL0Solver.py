import numpy as np
import pandas as pd
import torch
from L2Potts import L2Potts
class PottsL0Solver:
    def __init__(self, mImg, mWeights, mGamma):
        self.mImg = mImg
        self.mWeights = mWeights
        self.mGamma = mGamma
        self.mRow = mImg.shape[2]
        self.mCol = mImg.shape[3]
        self.mChannel = mImg.shape[1]
        
    def applyHorizontally(self):
        for i in range(0, self.mRow):
            l2potts = L2Potts(self.mImg[:,:,i,:], self.mWeights[i,:], self.mGamma)
            l2potts()
    def applyVertically(self):
        for j in range(0, self.mCol):
            l2potts = L2Potts(self.mImg[:,:,:,j], self.mWeights[:,j], self.mGamma)
            l2potts()
            
    def applyDiag(self):
        for k in range(0, self.mCol):
            l2potts = L2Potts(self.mImg.diagonal(offset = k, dim1=-2, dim2=-1), self.mWeights.diagonal(offset=k,dim1=-2,dim2=-1), self.mGamma)
            l2potts()
            
        for k in reversed(range(1, self.mRow)):
            l2potts = L2Potts(self.mImg.diagonal(offset = -k, dim1=-2, dim2=-1), self.mWeights.diagonal(offset=-k, dim1=-2,dim2=-1), self.mGamma)
            l2potts()
            
    def applyantiDiag(self):
        mImg_flipped = self.mImg.flip([0,2])
        mWeights_flipped = self.mWeights.flip(dims=(0,))
        
        for k in range(0, self.mCol):
            l2potts = L2Potts(mImg_flipped.diagonal(offset = k, dim1=-2, dim2=-1), mWeights_flipped.diagonal(offset=k,dim1=-2,dim2=-1), self.mGamma)
            l2potts()
            
        for k in reversed(range(1, self.mRow)):
            l2potts = L2Potts(mImg_flipped.diagonal(offset = -k, dim1=-2, dim2=-1), mWeights_flipped.diagonal(offset=-k, dim1=-2,dim2=-1), self.mGamma)
            l2potts()
        
        # self.mImg = mImg_flipped.flip([0,2])
        return mImg_flipped.flip([0,2])











