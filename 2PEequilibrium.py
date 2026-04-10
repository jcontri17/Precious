import numpy as np
import matplotlib.pyplot as plt
import math
#from functions4plasma import specificResistivity
#import functions4plasma.specificResistivity()
import functions4plasma


####################################################################### 
##########              FUNCTIONS & CONSTANTS                ##########
#######################################################################
###-----CONSTANTS
mu0 = np.pi * 4e-7          #permeability of free space [H/m]


###-----FUNCTION FOR MAGNETIC FIELD
def magField_2PE(r, Bs, Be, sig):
    bs = Bs / Be
    sig2 = (1 + 4*sig) * bs / (1 - bs)
    
    if np.isscalar(r):
        u = 2 * (r/Rs)**2 - 1
        
        #piecewise calculation for magnetic field [T]
        if np.abs(u) <= 1:
            B = Bs * u * np.exp(sig * (u**4 - 1))
        elif np.abs(u) > 1:
            B = Be - Be * (1 - bs) * np.exp(sig2 * (1 - u))
    else:
        nr = len(r)
        u = np.zeros(nr)
        B = np.zeros(nr)
        
        print("here")
        for i in range(nr):
            u[i] = 2 * (r[i]/Rs)**2 - 1
            #piecewise calculation for magnetic field [T]
            if np.abs(u[i]) <= 1:
                B[i] = Bs * u[i] * np.exp(sig * (u[i]**4 - 1))
            elif np.abs(u[i]) > 1:
                B[i] = Be - Be * (1 - bs) * np.exp(sig2 * (1 - u[i]))
                print(i)
    return B


###------FUNCTION FOR RADIAL GRADIANT OF MAGNETIC FIELD
def magFieldGrad_2PE(r, Bs, Be, sig):
    bs = Bs / Be
    sig2 = (1 + 4*sig) * bs / (1 - bs)
    
    if np.isscalar(r):
        u = 2 * (r/Rs)**2 - 1

        #piecewise calculation for radial gradiant of magnetic field [T/m]
        if np.abs(u) <= 1:
            dB__dr = (4 * Bs * r * np.exp(sig * (u**4 - 1)) / Rs**2) * (4 * sig * u**4 + 1)
        elif np.abs(u) > 1:
            dB__dr = (4 * r * Be * sig2 / Rs**2) * (1 - bs) * np.exp(sig2 * (1 - u))
    else:
        nr = len(r)
        u = np.zeros(nr)
        dB__dr = np.zeros(nr)
        
        for i in range(nr):
            u[i] = 2 * (r[i]/Rs)**2 - 1
            #piecewise calculation for radial gradiant of magnetic field [T/m]
            if np.abs(u[i]) <= 1:
                dB__dr[i] = (4 * Bs * r[i] * np.exp(sig * (u[i]**4 - 1)) / Rs**2) * (4 * sig * u[i]**4 + 1)
            elif np.abs(u[i]) > 1:
                dB__dr[i] = (4 * r[i] * Be * sig2 / Rs**2) * (1 - bs) * np.exp(sig2 * (1 - u[i]))
    return dB__dr



######################################################
##########              MAIN                ##########
######################################################
###-----INPUT PARAMETERS
bs = 0.75       #2PE parameter
sig = 0.2       #2PE parameter
Xs = 0.5        #FRC parameter

Be = 10         #external magnetic field [T]
Bs = bs * Be    #mangetic field at separatrix [T]
Rs = 0.002
n_e = 3e25      #number density [1/m^3]
T_e = 500       #plasma temperature [K]
Rc = Rs / Xs    #conducting wall radius [m]


del_r = Rc/100.0
r = np.arange(10*del_r, Rc+del_r, del_r)


B = magField_2PE(r, Bs, Be, sig)
dB__dr = magFieldGrad_2PE(r, Bs, Be, sig)
LB = B / dB__dr
eta = functions4plasma.specificResistivity(n_e, T_e, "eV")
tau_diff = mu0 * LB**2 / eta





######################################################
##########              MAIN                ##########
######################################################
labelFontSize = 14
titleFontSize = 18
textFontSize = 12

###-----COMPARISON OF NORMALIZED HALL AND PERPENDICULAR CONDUCTIVITIES
plt.figure(figsize=(8,8))
plt.plot(r/Rs, B/Be)
_2PEparams = "$b_s = {:.2f} = B(R_s)/B_{{ext}}$ \n $\sigma = {:.2f}$".format(bs, sig)
plt.text(0.7,0.4, _2PEparams, fontsize=textFontSize, transform=plt.gca().transAxes)

plt.xlabel('$r/R_s [m]$', fontsize=labelFontSize)
plt.ylabel('$B(r)/B_{ext}$', fontsize=labelFontSize)
figTitle = 'Normalized Magnetic Field\nFRC 1D 2PE Model'
plt.title(figTitle, fontsize=titleFontSize)
plt.grid()
plt.savefig(figTitle + '.png')
plt.show()



###-----COMPARISON OF NORMALIZED HALL AND PERPENDICULAR CONDUCTIVITIES
plt.figure(figsize=(8,8))
plt.plot(r/Rs, tau_diff)
_2PEparams = "$b_s = {:.2f} = B(R_s)/B_{{ext}}$ \n $\sigma = {:.2f}$".format(bs, sig)
plt.text(0.7,0.4, _2PEparams, fontsize=textFontSize, transform=plt.gca().transAxes)

plt.yscale('log')
plt.xlabel('$r/R_s [m]$', fontsize=labelFontSize)
plt.ylabel('$B(r)/B_{ext}$', fontsize=labelFontSize)
figTitle = 'Diffusion Time Scale\nFRC 1D 2PE Model'
plt.title(figTitle, fontsize=titleFontSize)
plt.grid()
plt.savefig(figTitle + '.png')
plt.show()












