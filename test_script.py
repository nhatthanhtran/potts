# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 22:25:32 2023

@author: jerrr
"""


import numpy as np
import pandas as pd
import torch
from PottsL0Solver import PottsL0Solver

test_torch1 = torch.randn(1,3, 200, 250)
test_weights = torch.randn(200, 250)
test_gamma = 1

img = PottsL0Solver(test_torch1, test_weights, test_gamma)