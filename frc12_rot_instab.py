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





t0code12 = time.perf_counter()

t1code12 = time.perf_counter()

tcode12 = t1code12 - t0code12
##########################################################################
#####                           OUTPUT                               #####
##########################################################################
print("\n\n")
print("FRC_12: Inputs and Domain =========================================")
print("COMPUTATIONAL METRICS---------------------")
print("\t frc12 Compute Time = {:.3e} [s]".format(tcode12))
