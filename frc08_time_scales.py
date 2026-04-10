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
from frc00_inputs_domain import *



t0code07 = time.perf_counter()


Te = T*11604

a = Rs / 2  #import this (torus minor radius)
D_Bohm = kb * Te / (16 * ee * Be)       #Bohm diffusion [m^2/s]
tau_Bohm = (mu0 * a**2) / (D_Bohm) 

om_pe = 


t1code07 = time.perf_counter()
tcode07 = t1code07 - t0code07
##########################################################################
#####                           OUTPUT                               #####
##########################################################################
print("\n\n")
print("FRC_00: Inputs and Domain =========================================")
print("TIME SCALE DEPENDENT PARAMETERS-----------")

print("\t D_Bohm = {:.3f}".format(D_Bohm))



print("COMPUTATIONAL METRICS---------------------")
print("\t frc07 Compute Time = {:.3e} [s]".format(tcode07))
