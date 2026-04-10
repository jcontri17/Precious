#2D FRC Equilibirium Input Geometries and Paremeters    Jeffrey D. Contri; jcontri@uw.edu 2025_09_28

##########################################################################
#####                           IMPORTS                              #####
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import cmath
import sys
import time
import re
from plottingParameters import *
from functions4plasma import *
from precious_functions import read_precious_settings




t0code00 = time.perf_counter()
##########################################################################
#####                           FUNCTIONS                            #####
##########################################################################




##########################################################################
#####                   FRC INITIAL CONDITIONS                       #####
##########################################################################
parameters = ["domdim", "model", "E", "Xs", "B0", "T", "Rw", "zLen", "zBuff", "XH", "XD", "XT", "Zi", "Nr", "Np", "Nz", "domx", "domy"]
myring = read_precious_settings("myPrecious.txt", required=parameters)
Lconv = (1e3)           #converts length parameters between SI input units (m) and practical units (mm)


B0          = myring["B0"]              #applied magnetic field (aka external coil field) [T]
T           = myring["T"]               #initial temperature [eV]
E           = myring["E"]               #FRC elongation [\]
Xs          = myring["Xs"]              #normalized separatrix radius [\]
Rw          = myring["Rw"]/Lconv        #liner radius [m]
zLen        = myring["zLen"]/Lconv      #liner length [m]
zBuff       = myring["zBuff"]           #ratio of X-point location to top edge of domain [\]

Rs          = Xs * Rw                   #separatrix radius [m]
Zs          = E * Rs                    #separatrix half-length [m]
Be          = B0 / (1 - Xs**2)          #external midplane magnetic field [T]

a           = Rs / 2                    #torus minor radius [m] (still do research on this)
a           = Rs
b           = Zs                        #torus half-length [m] (still do research on this)
kap         = b / a                     #torus elonation [\]

mH          = 1.007*amu                 #mass of proton [kg]
mD          = 2.014*amu                 #mass of deuterium [kg]
mT          = 3.016*amu                 #mass of tritium [kg]
XH          = myring["XH"]              #concentration of hydrogen
XD          = myring["XD"]              #concentration of deuterium
XT          = myring["XT"]              #concentration of tritium

m_ave       = (XH*mH + XD*mD + XT*mT)   #average ion mass [kg]
Zi          = myring["Zi"]              #ion charge number [\]



##########################################################################
#####                           DOMAIN                               #####
##########################################################################
domdim      = myring["domdim"]          #domain dimensions
model       = myring["model"]           #model for FRC domain (1D: 2PE; 2D: stein, sporer, jeff; 3D: NA)


###-----Constructing for (0<r, +/-z) and then reflecting for plotting
Rmax        = Rw                        #maximum r-domain value [m]
Rmin        = 0                         #minimum r-domain value [m]
Nr          = myring["Nr"]              #number of points in r-direction [\]
dr          = (Rmax - Rmin) / Nr        #radial differential [m]

Pmax        = 0.0 * np.pi/180           #maximum phi-domain value [rad] (float is in degrees)
Pmin        = 0.0 * np.pi/180           #minimum phi-domain value [rad] (float is in degrees)
Np          = myring["Np"]              #number of points in phi-direction [\]
dp          = (Pmax - Pmin) / Np        #azimuthal differential [rad]

Zmax        = zLen / 2                  #maximum z-domain value [m]
Zmin        = -zLen / 2                 #minimum z-domain value [m]
Nz          = myring["Nz"]              #number of points in z-direction [\]
dz          = (Zmax - Zmin) / Nz        #axial differential [m]



###----Calculating total number of grid points
if dr==0:
    Nr = 1
elif dp==0:
    Np = 1
elif dz==0:
    Nz = 1
Ntot = Nr*Np*Nz                         #total number of grid points [\]


domx        = myring["domx"]            #domain multiplier to make plotting modifications easier
domy        = myring["domy"]            #domain multiplier to make plotting modifications easier
Rd          = Rmax * domx               #domain radius [m]
#Pd = no need to expand the domain for visual purposes here
Zd          = Zmax * domy               #domain z-length [m]


###-----Constructing the Domain
if domdim=='1D':
    r = np.arange(dr, Rd+dr, dr)

elif domdim=='2D':
    r = np.arange(-Rd, Rd+dr, dr)
    z = np.arange(-Zd, Zd+dz, dz)    
    R, Z = np.meshgrid(r, z, indexing='ij')
    Rcc = 0.5 * (R[:-1, :-1] + R[1:, 1:])
    Zcc = 0.5 * (Z[:-1, :-1] + Z[1:, 1:])

elif domdim=='3D':
    r = np.arange(-Rd, Rd+dr, dr)
    p = np.arange(Pmin, Pmax+dp, dp)
    z = np.arange(-Zd, Zd+dz, dz)

else:
    print("ERROR: Set domdim to '1D', '2D', or '3D'.")





t1code00 = time.perf_counter()
##########################################################################
#####                           OUTPUT                               #####
##########################################################################
Ltab = 20

summary_text = f"""
FRC_00: Inputs and Domain =============================================
INITIALIZATION------------------------------
    E  = {f'{E:.3f}':<{Ltab}} - FRC Elongation
    Xs = {f'{Xs:.3f}':<{Ltab}} - Normalized Separatrix Radius
    Rw = {f'{Rw*Lconv:.3f} [mm]':<{Ltab}} - Wall Radius
    Rs = {f'{Rs*Lconv:.3f} [mm]':<{Ltab}} - Separatrix Radius
    Zs = {f'{Zs*Lconv:.3f} [mm]':<{Ltab}} - Half Length
    B0 = {f'{B0:.3f} [T]':<{Ltab}} - External Field
    Be = {f'{Be:.3f} [T]':<{Ltab}} - Equilibrium Field
    T  = {f'{T:.3f} [eV]':<{Ltab}} - ({T*11604:.3f} K)
    m_i = {f'{m_ave:.3e} [kg]':<{Ltab}} - average ion mass
    Zi  = {f'{Zi:.0f}':<{Ltab}} - average ion charge
DOMAIN---------------------------------------
    domdim = {domdim}
    model  = {model}
    r-domain: {f'{Rmin*Lconv:.3f}'} < r < {f'{Rmax*1e3:.3f}'} [mm]    dr = {dr*Lconv:.3f} [mm]
    z-domain: {f'{Zmin*Lconv:.3f}'} < z < {f'{Zmax*1e3:.3f}'} [mm]    dz = {dz*Lconv:.3f} [mm]
    Total Grid Points = {Ntot:.0f}
        Nr = {Nr:.0f}
        Nφ = {Np:.0f}
        Nz = {Nz:.0f}
COMPUTATIONAL METRICS------------------------
    frc00 Compute Time = {t1code00-t0code00:.3e} [s]

    
"""



with open("adventure_log.txt", "w") as alog:
    alog.write(summary_text)






















