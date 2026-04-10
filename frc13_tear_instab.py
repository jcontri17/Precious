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





t0code13 = time.perf_counter()

t1code13 = time.perf_counter()

tcode13 = t1code13 - t0code13
##########################################################################
#####                           OUTPUT                               #####
##########################################################################
print("\n\n")
print("FRC_13: Inputs and Domain =========================================")
print("COMPUTATIONAL METRICS---------------------")
print("\t frc13 Compute Time = {:.3e} [s]".format(tcode13))
