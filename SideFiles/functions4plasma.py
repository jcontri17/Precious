import numpy as np
import matplotlib.pyplot as plt
import math
import sys

eV = 1.6022e-19
kb = 1.38e-23
amu = 1.66e-27
Rbar = 8.314        #J/mol/K
ee = 1.6022e-19
m_e = 5.486e-4*amu
mu0 = np.pi * 4e-7
eps0 = 8.85e-12

#source: DOI 10.1007/978-1-4757-4030-1 (pg 10)
def plasmaFreq(n0, m):
    #[n0] = m^-3
    #[m] = kg
    ee = 1.6022e-19     #elementary charge [C]
    eps0 = 8.85e-12     #permittivity of free space [F/m]

    om_p = np.sqrt((n0 * ee**2) / (m * eps0))     #plasma frequency [rad/s]
    return om_p



#source: DOI 10.1063/1.3613680 (sec II.B)
def skinDepth(n0, m):
    #[n0] = m^-3
    #[m] = kg
    c = 2.98e8                  #speed of light [m/s]

    om_p = plasmaFreq(n0, m)    #plasma frequency [rad/s]

    delt_skin = c / om_p        #skin depth (ion or electron) [m]
    return delt_skin



#source: DOI 10.1007/978-1-4757-4030-1 (pg 8)
def debyeLength(n_e, T, T_units):
    #[n_e] = m^-3
    kb = 1.38e-23       #Boltzmann constant [J/K]
    eV = 1.6022e-19     #conversion from eV to J [J/eV]
    eps0 = 8.85e-12     #permittivity of free space [F/m]
    ee = 1.6022e-19     #elementary charge [C]

    if T_units == "K":
        T_e = T
    elif T_units == "eV":
        T_e = T * eV / kb
    else:
        raise ValueError(f"Unsupported units for Temperature: {T_unit}")

    lamb_D = np.sqrt((eps0 * kb * T_e) / (n_e * ee**2))     #Debye Length [m]
    return lamb_D

#source: DOI 10.1007/978-1-4757-4030-1 (pg 377)
def AlfvenSpeed(B0, rho0):
    #[B0] = T
    #[rho0] = kg/m^3
    mu0 = np.pi * 4e-7                  #permeability of free space [H/m] or [kg*m^2/s^2/A^2/m]

    v_A = B0 / np.sqrt(mu0 * rho0)      #Alfven Speed [m/s]
    return v_A



#source: DOI 10.1007/978-3-319-22309-4 (eq 5.72)
def LamLogChen(n_e, T, T_units):
    #[n_e] = m^-3
    kb = 1.38e-23       #Boltzmann constant [J/K]
    eV = 1.6022e-19     #conversion from eV to J [J/eV]

    if T_units == "K":
        T_e = T
    elif T_units == "eV":
        T_e = T * eV / kb
    else:
        raise ValueError(f"Unsupported units for Temperature: {T_unit}. Choose K or eV.")
    
    lamb_D = debyeLength(n_e, T, T_units)               #Debye-Huckel length [m]

    LamLog = math.log(12 * np.pi * n_e * lamb_D**3)     #Coulomb logarithm [\]
    return LamLog


#source: DOI 10.1002/3527607978 (pg 182)
def LamLogTokamak(n_e, T, T_units):
    kb = 1.38e-23       #Boltzmann constant [J/K]
    eV = 1.6022e-19     #conversion from eV to J [J/eV]
    
    if T_units == "K":
        T_e = T
    elif T_units == "eV":
        T_e = T * eV / kb
    else:
        raise ValueError(f"Unsupported units for Temperature: {T_unit}. Choose K or eV.")

    if T_e < 1.16e5:
        LamLog = 16.34 + 1.5*math.log(T_e) - 0.5*math.log(n_e)
    elif T_e > 1.16e5:
        LamLog = 22.81 + math.log(T_e) - 0.5*math.log(n_e)

    return LamLog




#source: DOI 10.1007/978-1-4757-4030-1 (pg 39)
def cycloFreq(B, Z, m):
    #[B] = T
    #[m] = kg
    #[Z] = none
    ee = 1.6022e-19         #elementary charge [C]

    om_c = Z * ee * B / m   #cyclotron frequency [rad/s]
    return om_c



#source: DOI 10.1007/978-3-319-22309-4 (eq 5.71)
def specificResistivity(n_e, T, T_units):
    #[n_e] = m^-3
    ee = 1.6022e-19     #elementary charge [C]
    m_e = 9.11e-31      #electron mass [kg]
    eps0 = 8.85e-12     #permittivity of free space [F/m] or [C^2*s^2/kg/m^2/m]
    kb = 1.38e-23       #Boltzmann constant [J/K]
    eV = 1.6022e-19     #conversion from eV to J [J/eV]

    if T_units == "K":
        T_e = T
    elif T_units == "eV":
        T_e = T * eV / kb
    else:
        raise ValueError(f"Unsupported units for Temperature: {T_unit}. Choose K or eV.")
    LamLog = LamLogTokamak(n_e, T_e, "K")       #Coulomb logarithm [\]
    
    eta = (np.pi * ee**2 * np.sqrt(m_e)) / ((4*np.pi*eps0)**2 * (kb*T_e)**1.5) * LamLog     #resisitivity [ohm*m] or [kg*m^2/C^2/s * m]
    return eta




#source: DOI 10.1007/978-3-319-22309-4 (eq 5.62)
def collFreq(n_e, T, T_units):
    ee = 1.6022e-19     #elementary charge [C]
    m_e = 9.11e-31      #electron mass [kg]
    kb = 1.38e-23       #Boltzmann constant [J/K]
    eV = 1.6022e-19     #conversion from eV to J [J/eV]

    if T_units == "K":
        T_e = T
    elif T_units == "eV":
        T_e = T * eV / kb
    else:
        raise ValueError(f"Unsupported units for Temperature: {T_unit}. Choose 'K' or 'eV'.")
    eta = specificResistivity(n_e, T_e, T_units)  #specific resisitivity [ohm*m]

    nu_ei = (n_e * ee**2) * eta / m_e        #collision frequency [#/s]
    return nu_ei


#source: ?
def HallParameter(B, n_e, T, T_units):
    #[B] = T
    #[n_e] = m^-3
    m_e = 9.11e-31      #electon mass [kg]   
    kb = 1.38e-23       #Boltzmann constant [J/K]
    eV = 1.6022e-19     #conversion from eV to J [J/eV]
 
    if T_units == "K":
        T_e = T
    elif T_units == "eV":
        T_e = T * eV / kb
    else:
        raise ValueError(f"Unsupported units for Temperature: {T_unit}. Choose 'K' or 'eV'.")

    om_ce = cycloFreq(B, 1, m_e)                    #electron cyclotron frequency [1/s]
    nu_ei = collFreq(n_e, T, T_units)    #collision frequency [1/s]    
    tau_ei = 1 / nu_ei                              #average collision time [s]
    HP = om_ce * tau_ei                             #Hall parameter [\]
    
    HP_params = [HP, om_ce, tau_ei]
    return HP_params


#source: ?
def meanFreePath(n0, T, T_units):
    kb = 1.38e-23       #Boltzmann constant [J/K]
    eV = 1.6022e-19     #conversion from eV to J [J/eV]
    
    if T_units == "K":
        T_e = T
    elif T_units == "eV":
        T_e = T * eV / kb
    else:
        raise ValueError(f"Unsupported units for Temperature: {T_unit}. Choose 'K' or 'eV'.")
    
    N_D = debyeSphere(n0, T, T_units)
    lamb_D = debyeLength(n0, T, T_units)        #Debye-Huckel Length [m]

    lamb_mfp = N_D * lamb_D                     #mean free path [m]
    return lamb_mfp


#source: DOI 10.1007/978-1-4757-4030-1 (pg 8)
def debyeSphere(n0, T, T_units):
    eV = 1.6022e-19     #conversion from eV to J [J/eV]
    kb = 1.38e-23       #Boltzmann constant [J/K]
    
    if T_units == "K":
        T_e = T
    elif T_units == "eV":
        T_e = T * eV / kb
    else:
        raise ValueError(f"Unsupported units for Temperature: {T_unit}. Choose 'K' or 'eV'.")
    
    lamb_D = debyeLength(n0, T, T_units)        #Debye-Huckel length [m]

    N_D = (4/3) * np.pi * n0 * lamb_D**3        #number of particles in a Debye Sphere [#]
    return N_D


#source: ?
def gyroRadius(m, v_perp, Z, B):
    ee = 1.6022e-19     #elementary charge [C]

    r_L = m * v_perp / (Z * ee * B)
    return r_L






def thermalVelocity(m, T, T_units):
    eV = 1.6022e-19     #conversion from eV to J [J/eV]
    kb = 1.38e-23       #Boltzmann constant [J/K]
    
    if T_units == "K":
        T = T
    elif T_units == "eV":
        T = T * eV / kb
    else:
        raise ValueError(f"Unsupported units for Temperature: {T_unit}. Choose 'K' or 'eV'.")

    v_th = np.sqrt(kb * T / m)                  #thermal speed [m/s]
    return v_th















