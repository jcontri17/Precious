#2D FRC Equilibirium Input Geometries and Paremeters    Jeffrey D. Contri; jcontri@uw.edu 2025_09_28

#################################################
#####               IMPORTS                 #####
#################################################
import numpy as np
import matplotlib.pyplot as plt
import cmath
import sys
import time
from plottingParameters import *
from functions4plasma import *





t0code01 = time.perf_counter()

t1code01 = time.perf_counter()

tcode01 = t1code01 - t0code01
##########################################################################
#####                           OUTPUT                               #####
##########################################################################
print("\n\n")
print("FRC_01: 1-DIMENSIONAL MODEL =========================================")
print("COMPUTATIONAL METRICS---------------------")
print("\t frc01 Compute Time = {:.3e} [s]".format(tcode01))
