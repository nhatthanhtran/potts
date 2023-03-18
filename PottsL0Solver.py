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
			self.mImg[:,:,i,:] = L2Potts(self.mImg[:,:,i,:], self.mWeights[i,:], mGamma)

	def applyVertically(self):
		for j in range(0, self.mCol):
			self.mImg[:,:,:,j] = L2Potts(self.mImg[:,:,:,j], self.mWeights[:,j], mGamma)

	def applyDiag(self):
		






