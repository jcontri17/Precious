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


t0code06 = time.perf_counter()

t1code06 = time.perf_counter()

tcode06 = t1code06 - t0code06
##########################################################################
#####                           OUTPUT                               #####
##########################################################################
print("\n\n")
print("FRC_06: Inputs and Domain =========================================")
print("COMPUTATIONAL METRICS---------------------")
print("\t frc06 Compute Time = {:.3e} [s]".format(tcode06))



