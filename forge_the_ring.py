#AA 560 - Homework 5            Jeffrey D. Contri; jcontri@uw.edu 2026_03_10

##########################################################################
#####                           IMPORTS                              #####
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import cmath
import re
import sys
import time
from plottingParameters import *
from functions4plasma import *
from precious_functions import read_precious_settings



##########################################################################
#####                           FUNCTIONS                            #####
##########################################################################





##########################################################################
##########################################################################
##########                          MAIN                       ###########
##########################################################################
##########################################################################
# regex: variable_name = value  (whitespace-insensitive)
_LINE_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$')
parameters = ["domdim", "model"]
myring = read_precious_settings("myPrecious.txt", required=parameters)


domdim  = myring["domdim"]
model   = myring["model"]



import frc00_inputs_domain

if domdim == '1D':
    if model == '2PE':
        import frc01_1D_2PE
elif domdim == '2D':
    if model == 'RR':
        pass
    elif model == 'stein':
        import frc04_2D_Steinhauer
    elif model == 'cfc':
        import frc02_GSderivation
    elif model == 'sporer':
        import frc04_2D_Steinhauer
elif domdim == '3D':
    pass

    
























