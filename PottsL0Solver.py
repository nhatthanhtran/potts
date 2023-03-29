import numpy as np
import pandas as pd
import torch

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
			L2Potts(self.mImg[:,:,i,:], self.mWeights[i,:], self.mGamma)

	def applyVertically(self):
		for j in range(0, self.mCol):
			L2Potts(self.mImg[:,:,:,j], self.mWeights[:,j], self.mGamma)

	def applyDiag(self):
		for k in range(0, self.mCol):
			L2Potts(self.mImg.diagonal(offset = k, dim1=-2, dim2=-1), self.mWeights(offset=k,dim1=-2,dim2=-1), self.mGamma)

		for k in reversed(range(1, self.mRow)):
			L2Potts(self.mImg.diagonal(offset = -k, dim1=-2, dim2=-1), self.mWeights(offset=-k, dim1=-2,dim2=-1), self.mGamma)

	def applyantiDiag(self):
		mImg_flipped = self.mImg.flip([0,2])
		mWeights_flipped = self.mWeights.flip(dims=(0,))

		for k in range(0, self.mCol):
			L2Potts(self.mImg_flipped.diagonal(offset = k, dim1=-2, dim2=-1), self.mWeights_flipped(offset=k,dim1=-2,dim2=-1), self.mGamma)











