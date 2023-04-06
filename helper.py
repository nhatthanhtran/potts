import torch
import numpy as np
from PottsL0Solver import PottsL0Solver
def minL2PottsADMM4(img, gamma, weights, muInit, mustep, stopTol, verbose, multiThreaded, useADMM):
    m, n, l = img.shape

    u = torch.zeros(m, n, l)
    v = img.clone()
    lam = torch.zeros(m, n, l)
    temp = torch.zeros(m, n, l)
    weightsPrime = torch.zeros(m, n)
    error = np.inf
    mu = muInit
    gammaPrime = 0.0
    nIter = 0
    fNorm = 0.1 #PLACE HOLDER NEED TO CHANGE img.normQuad()

    if fNorm == 0:
        return img

    while (error >= stopTol * fNorm):
        # set Potts parameters
        gammaPrime = 2*gamma

        #set weights
        weightsPrime += mu 

        #solve horizontal univariate Potts problems
        #COULD USE BROADCAST
        for i in range(m):
            for j in range(n):
                for k in range(l):
                    u[i,j,k] = (img[i,j,k]*weights[i,j] + v[i,j,k]*mu - lam[i,j,k]) / weightsPrime[i,j]
        
        #THIS COULD BE PROBLEMATIC BECAUSE WE WANT V TO CHANGE?
        horizontal_proc = PottsL0Solver(u, weightsPrime, gammaPrime)
        horizontal_proc.applyHorizontally()

        #solve for vertical univariate Potts problems
        #COULD USE BROADCAST
        for i in range(m):
            for j in range(n):
                for k in range(l):
                    v[i,j,k] = (img[i,j,k]*weights[i,j] + u[i,j,k]*mu + lam[i,j,k]) / weightsPrime[i,j]

        #THIS COULD BE PROBLEMATIC BECAUSE WE WANT V TO CHANGE?
        vertical_proc = PottsL0Solver(v, weightsPrime, gammaPrime)
        vertical_proc.applyVertically()

        #update Lagrange multiplier and calculate difference between u and v
        #COULD USE BROADCAST
        error = 0
        for i in range(m):
            for j in range(n):
                for k in range(l):
                    temp[i,j,k] = u[i,j,k] - v[i,j,k]
        
                    if useADMM:
                        lam[i,j,k] = lam[i,j,k] + temp[i,j,k]
                    
                    error += torch.pow(temp[i,j,k],2)

        #update coupling
        mu *= mustep

        #count iterations
        nIter +=1

        #show information
        if verbose:
            print("*")
            if nIter%50 == 0:
                print("\n")

    return u
        












    return 0