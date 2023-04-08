# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:25:32 2023

@author: jerrr
"""


import numpy as np
import pandas as pd
import torch
from PottsL0Solver import PottsL0Solver
from helper import minL2PottsADMM4
test_torch1 = torch.randn(1,3, 200, 250)
test_weights = torch.randn(200, 250)
test_gamma = 1

# img = PottsL0Solver(test_torch1, test_weights, test_gamma)
# img.applyHorizontally()
# img.applyVertically()
# img.applyDiag()
# img1 = img.applyantiDiag()
# a = 1

img2 = minL2PottsADMM4(test_torch1, test_gamma, test_weights, 1, 0.1, 0.1, False, False, True)
