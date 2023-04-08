import torch
import numpy as np
from PottsL0Solver import PottsL0Solver

def minL2PottsADMM4(img, gamma, weights, muInit, mustep, stopTol, verbose, multiThreaded, useADMM, omega):
    b, l, m, n  = img.shape

    u = torch.zeros(b, l, m, n)
    v = img.clone()
    w = img.clone()
    z = img.clone()
    lam1 = torch.zeros(b, l, m, n)
    lam2 = torch.zeros(b, l, m, n)
    lam3 = torch.zeros(b, l, m, n)
    lam4 = torch.zeros(b, l, m, n)
    lam5 = torch.zeros(b, l, m, n)
    lam6 = torch.zeros(b, l, m, n)
    # temp = torch.zeros(l, m, n)
    weightsPrime = torch.zeros(m, n)
    error = np.inf
    mu = muInit
    gammaPrimeC = 0.0
    gammaPrimeD = 0.0
    omegaC, omegaD = omega

    nIter = 0
    fNorm = torch.sum(torch.pow(img, 2)) #PLACE HOLDER NEED TO CHANGE img.normQuad()

    if fNorm == 0:
        return img

    while (error >= stopTol * fNorm):
        # set Potts parameters
        gammaPrimeC = 4.0*omegaC*gamma
        gammaPrimeD = 4.0*omegaD*gamma

        #set weights
        weightsPrime = weights + 6*mu

        #solve horizontal univariate Potts problems
        u = torch.divide((torch.multiply(img, weights) + 2*mu*(w + v + z) + 2*(-lam1 - lam2 - lam3)), weightsPrime)
        #THIS COULD BE PROBLEMATIC BECAUSE WE WANT U TO CHANGE?
        horizontal_proc = PottsL0Solver(u, weightsPrime, gammaPrimeC)
        horizontal_proc.applyHorizontally()

        #solve 1D Potts problems diagonally
        w = torch.divide(torch.multiply(img, weights) + 2*mu*(u+v+z) + 2*(lam2 + lam4 -lam6), weightsPrime)
        diagonal_proc = PottsL0Solver(w, weightsPrime, gammaPrimeD)
        diagonal_proc.applyDiag()

        #solve for vertical univariate Potts problems
        v = torch.divide((torch.multiply(img, weights) + 2*mu*(u+w+z) + 2*(lam1-lam4-lam5)), weightsPrime)
        
        #THIS COULD BE PROBLEMATIC BECAUSE WE WANT V TO CHANGE?
        vertical_proc = PottsL0Solver(v, weightsPrime, gammaPrimeC)
        vertical_proc.applyVertically()

        #solve 1D Potts problems antidiagonally
        z = torch.divide(torch.multiply(img, weights) + 2*mu*(u+w+v) + 2*(lam3+lam5+lam6), weightsPrime)
        antidiag_proc = PottsL0Solver(z, weightsPrime, gammaPrimeD)
        z = antidiag_proc()

        #update Lagrange multiplier and calculate difference between u and v

        error = torch.sum(torch.pow(u-v, 2), dim=None)
        if useADMM:
            lam1 = lam1 + mu*(u-v)
            lam2 = lam2 + mu*(u-w)
            lam3 = lam3 + mu*(u-z)
            lam4 = lam4 + mu*(v-w)
            lam5 = lam5 + mu*(v-z)
            lam6 = lam6 + mu*(w-z)

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