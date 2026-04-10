#FRC 2D Equilibrium         Jeffrey D. Contri; jcontri@uw.edu           2025_04_30 - 2025_08_01
#ψ

#################################################
#####               IMPORTS                 #####
#################################################
import numpy as np
import matplotlib.pyplot as plt
import cmath
import sys
import time
from functions4plasma import *
from plottingParameters import *
from scipy.optimize import fsolve
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from matplotlib.ticker import AutoMinorLocator
from matplotlib.path import Path
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LogNorm
from matplotlib import colors as mcolors
#from scipy.integrate import cumulative_trapezoid as cumtrapz
import frc00_inputs_domain as frc00
from frc05_2D_SteinhauerFunctions import *
from forge_the_ring import read_precious_settings


t0code = time.perf_counter()
#################################################
#####               FUNCTIONS               #####
#################################################





#######################################################################################################
#######################################################################################################
#################################                  MAIN                    ############################
#######################################################################################################
#######################################################################################################
parameters = ["sig", "f"]
myring = read_precious_settings("myPrecious.txt", required=parameters)
Lconv = 1e-2            #length conversion used so that displayed numerical lengths are in [cm]

###------------------------------------------------------------------------------------------PARAMETERS
B0          = frc00.B0                      #applied coil magnetic field [T]
T           = frc00.T                       #Temperature [eV]
E           = frc00.E                       #FRC elongation [\]
Xs          = frc00.Xs                      #normalized separatrix radius [\]
zLen        = frc00.zLen                    #z-length of liner [cm]
Zi          = frc00.Zi                      #ion charge number [\]

h           = zLen / 2                      #liner height (from z=0) [m]
liner_top   = h                             #liner top [m]
liner_bot   = -h                            #liner bottom [m]

Rc          = frc00.Rw                      #coil radius; assumed to by Rw here [m]
Rw          = frc00.Rw                      #flux-conserving wall radius [m]
Rs          = frc00.Rs                      #separatrix radius [m]
Zs          = frc00.Zs                      #separatrix half-length [m]


a           = frc00.a                       #FRC semi-minor radius [m]
b           = frc00.b                       #FRC semi-major radius [m]
sig         = myring["sig"]                 #flare parameter; adjustable parameter that's fixed for Steinhauer's paper
f           = myring["f"]                   #internal psi error factor for Sporer's approximation
eps         = a / b                         #inverse elongation

Be          = frc00.Be                      #external magnetic field [T]
Bw          = Be                            #wall magnetic field [T]
m_ave       = frc00.m_ave                   #average ion mass [kg]
TK          = T * eV/kb                     #Temp number in [K]

zBuff       = frc00.zBuff
if Zs > zBuff*h:
    print("FRC is too long: Zs > zBuff*h")
    print(" {:.2e} [m] > {:.2e} [m]".format(Zs, zBuff*h))
    print(" Reduce E or Xs")



###----------------------------------------------------------------------------------------DOMAIN SETUP
"""
domx = 1.0                                  #weight to extend the domain; helps show psi=1 curves better
Rd = Rw * domx                              #half-length of computational domain in r-dir [m]
Zd = h * domx                             #half-length of computational domain in z-dir [m]

dr = 0.001*Lconv                            #mesh fidelity in r-dir [m]
dz = 0.001*Lconv                            #mesh fidelity in z-dir [m]

r = np.arange(-Rd, Rd + dr, dr)
z = np.arange(-Zd, Zd + dz, dz)
R, Z = np.meshgrid(r, z, indexing='ij')

R_cc = 0.5 * (R[:-1, :-1] + R[1:, 1:])
Z_cc = 0.5 * (Z[:-1, :-1] + Z[1:, 1:])
"""

Rd = frc00.Rd
Zd = frc00.Zd
r = frc00.r
z = frc00.z
R = frc00.R
Z = frc00.Z
Rcc = frc00.Rcc
Zcc = frc00.Zcc




###-------------------------------------------------------------------------------------EXTERNAL REGION
Eguess = [Be, Be/3, Be/5, 0.9]              #initial guess for E parameters
sol = root(                                 #solves the system of external E equations
    fun=external_E_params,                  #designates the function
    x0=Eguess,                              #initial guesses for E0, E1, E2, alpha
    args=(eps, Xs, sig),                    #designates these parameters as constants
    method='hybr',                          #same algorithm as old fsolve, or try 'lm'
    tol=1e-6                                #tolerance
)
if not sol.success:                         #if E root finder is unsuccessful, display error message
    print("root() failed to converge:", sol.message)
    Br_ext, Bz_ext = 0
else:
    E0, E1, E2, alpha = sol.x               #grabs the solutions for E0, E1, E2, alpha

    psi_ext = external_psi(R, Z, Be, a, b, E0, E1, E2, alpha)                  #external magnetic flux
    dpsi__dr_ext = external_dpsi__dr(R, Z, Bw, a, b, E0, E1, E2, alpha)         #[T*m^2]; gradient of
    dpsi__dz_ext = external_dpsi__dz(R, Z, Bw, a, b, E0, E1, E2, alpha)         #external magnetic flux
    Br_ext = -(1/R) * dpsi__dz_ext          #radial magnetic field [T]          #[T*m]
    Bz_ext = (1/R) * dpsi__dr_ext           #axial magnetic field [T]
    #check ^these^ signs

N = shape_index_N3(a, b, Xs)                #shape index [\]; N=0 (racetrack), N=1 (ellipse/Hills vortex)
D0 = internal_D0(eps)                       #D_0 constant [\]
D1 = internal_D1(eps)                       #D_1 constant [\]
b0 = internal_B0(a, b, Xs, Bw)              #nominal magnetic field [T]
b1 = internal_B1(b0, D0, D1, N, eps)        #"other field component"; not pictured [T]
Beplus = external_Be(Bw, a, b, Xs)              #External magnetic field just outside the separatrix [T]


###-------------------------------------------------------------------------------------INTERNAL REGION
model = "sporer"     #choose between Steinhauer's full solution (stein), or the approx for E>>1 (sporer)

if model=="stein":
    psi_int = internal_psi_stein(R, Z, a, b, b0, b1, D0, D1)                #internal magnetic flux
    Br_int = (1/R) * internal_dpsi__dz_stein(R, Z, a, b, b0, b1, D0, D1)    #[T*m^2]; internal radial and
    Bz_int = -(1/R) * internal_dpsi__dr_stein(R, Z, a, b, b0, b1, D0, D1)   #axial magnetic fields [T]
elif model=="sporer":
    psi_int = internal_psi_sporer(R, Z, a, b, Bw, Xs, f)                   
    Br_int = (1/R) * internal_dpsi__dz_sporer(R, Z, a, b, Bw, Xs)
    Bz_int = -(1/R) * internal_dpsi__dr_sporer(R, Z, a, b, Bw, Xs, f)


###----------------------------------------------------------------MAGNETIC FIELD CONSTRUCTION - B(r,z)
#net or cutoff model for int/ext magnetic flux functions
#"net"      -> psi(r,z) = psi_ext(r,z) - psi_int(r,z)
#"cutoff"   -> uses the np.where function with some condition
construct = "cutoff"   


if construct=="cutoff":
    inside = (psi_int > 0)
    #inside = (psi_ext < 0)                      #condition for setting where the internal psi function  
    psi = np.where(inside, psi_int, psi_ext)    #is applied to create a full psi(r,z) [T*m^2]
    Br = np.where(inside, Br_int, Br_ext)       #Full radial magnetic field profile, Br(r,z) [T]
    Bz = np.where(inside, Bz_int, Bz_ext)       #Full axial magnetiic field profiel, Bz(r,z) [T]
elif construct=="net":
    psi = psi_ext - psi_int
    dpsi_dr, dpsi_dz = np.gradient(psi, R, Z, edge_order=2)
    Br = -(1/R) * dpsi_dz
    Bz = (1/R) * dpsi_dr
# Br(r,z) and Bz(r,z) have been created


B_int = np.sqrt(Br_int**2 + Bz_int**2)
B_ext = np.sqrt(Br_ext**2 + Bz_ext**2)

Bmag = np.sqrt(Br**2 + Bz**2)                       

ave_Bi = np.mean(B_int)
Bmax = np.max(B_int)





###----------------------------------------------------------------SEPARATRIX & MIDPLANE SLICE METRICS
rSep, zSep = get_flux_contours(psi_int, R, Z, 0)        #gets a contour of the separatrix (psi=0)
sepFile = "separatrix_rz.txt"
Lconv = 1e-3
np.savetxt(sepFile, np.column_stack([rSep/Lconv, zSep/Lconv]), 
           header="r [mm]        z [mm]", fmt="%.9f")

###-----Debugging for negative region between psi_ext and psi_int
# find every negative ψ
neg_vals = psi[psi < 0]


###-----Midplane Slices
mid_r = int(len(r)/2)
mid_z = int(np.argmin(np.abs(z)))       #grabs the index of z=0
mask = r >= 0                           #creates a mask to grab the r>=0 array values for lineout

# 1-D r and matching Bz along z=0
rSlice = r[mask]            # shape (Nr_pos,)
BSlice = Bz[mask, mid_z]    # shape (Nr_pos,)

psi_int_mask = (r >= 0) & (r <= Rs)
psi_ext_mask = (r > Rs) & (r <= Rw)
psiSlice = np.concatenate([psi_int[psi_int_mask, mid_z], psi_ext[psi_ext_mask, mid_z]])
rSlice_psi = np.concatenate([r[psi_int_mask], r[psi_ext_mask]])
psi_min = psi.min()























###-----Axes Placement
figXsize = 12
figYsize = 8

Xax1_start = 0.025
Yax1_start = 0.1
Xax1_len   = 0.25
Yax1_len   = 0.8

Xax2_start = 0.5
Yax2_start = 0.1
Xax2_len   = 0.4
Yax2_len   = 0.8





plot_idx = 0
##########=========================     PLOTTING ψ(r,z)     =========================##########
plot_idx = plot_idx + 1

figTitle = 'Flux Contour'
saveTitle = str(re.sub(r'[\\/*?:"<>|\n$]', '_', figTitle))
ax00_saveTitle = saveTitle
ax01_saveTitle = 'Flux Midplane Lineout'

cmapName = 'viridis'
labelFontWeight = 'bold'
lineWidth = 2
seplineWidth = 2
markerSize = 1

Lconv = 1e-3
psiL = 1e-2         #converts to cm making lineout graph axes nice
X = R/Lconv          # r in mm (cell centers)
Y = Z/Lconv          # z in mm
Zvar = psi
vmin = 0
vmax = axesLimit(psi,1)
numColors = 11
colors = plt.cm.rainbow(np.linspace(0, 1, numColors))
colors =  colors[::-1]
levels = 100*Lconv**2 * np.arange(0, 1.1, 0.1)       #defines contour levels at ψ = 0.0, 0.1, 0.2, …, 1.0
#levels = np.arange(-0.1, 1.2, 0.1)       #defines contour levels at ψ = 0.0, 0.1, 0.2, …, 1.0
handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(levels))]
labels  = [fr'$\psi={lev:.1e}$' for lev in levels]
labels = [r'$\boldsymbol{\psi}=$' + sci_label(lev, bold=True) for lev in levels]

###----------Ax1: Magnetic Field Contour
fig1 = plt.figure(figsize=(figXsize, figYsize))
ax00 = fig1.add_axes([Xax1_start, Yax1_start, Xax1_len, Yax1_len])     #[x, y, width, height]
ax01 = fig1.add_axes([Xax2_start, Yax2_start, Xax2_len, Yax2_len])     

# draw thin black contour lines at every 0.1
cs = ax00.contour(X, Y, Zvar,
                levels=levels,
                colors='k',
                linewidths=0.5)
"""
# fill between contours with a colormap
cf = ax.contourf(R/Lconv, Z/Lconv, psi,
                 levels=levels,
                 cmap=cmapName)
"""
ax00.contour(R/Lconv, Z/Lconv, psi,
           levels=levels,
           colors=[colors[i] for i in range(len(levels))],
           linewidths=lineWidth)
"""
# label each contour line with its ψ value
ax00.clabel(cs,
          fmt="%1.e",
          inline=True,
          fontsize=8)
"""
title_txt = fr'$\boldsymbol{{\varphi_{{\min}}}} = \boldsymbol{{{psi_min/psiL**2:.3g}}}\ \boldsymbol{{[T\cdot \boldsymbol{{cm}}^2]}}$'
ax00.legend(handles, labels, title=title_txt,
            loc='center left', bbox_to_anchor=(1.1, 0.5),
            frameon=False, borderaxespad=0.0)
ax00.plot(rSep/Lconv, zSep/Lconv, color='black', linestyle=':', linewidth=seplineWidth)


ax00.set_xlabel(r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$")
ax00.set_ylabel(r"$\boldsymbol{z}$  $\boldsymbol{[mm]}$")
ax00.tick_params(axis='both', labelsize=12)
for label in ax00.get_xticklabels() + ax00.get_yticklabels():
    label.set_fontweight('bold')
ax00.set_xlim(-Rd/Lconv, Rd/Lconv)
ax00.set_ylim(-Zd/Lconv, Zd/Lconv)
ax00.set_title(ax00_saveTitle, fontweight=labelFontWeight)
ax00.set_aspect('equal', adjustable='box')



###----------Ax2: Magnetic Field Midplane Lineout
X_array = rSlice_psi/Lconv
Y_array = psiSlice/psiL**2


X_min   = 0
X_max   = np.ceil(X_array[-1])
X_step  = 10**np.floor(np.log10(X_array[-1]))
Y_min   = -1
Y_max   = 2
Y_step  = 0.1

X_label = r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$"
Y_label = r"$\boldsymbol{\psi(r,0)}$  $\boldsymbol{[T \cdot \boldsymbol{cm}^2]}$"

ax01.plot(X_array, Y_array, linewidth=lineWidth)
ax01.set_xlabel(X_label)
ax01.set_ylabel(Y_label)
ax01.set_xlim(X_min, X_max)
ax01.set_ylim(Y_min, Y_max)
#ax01.set_xticks(np.arange(X_min, X_max+X_step, X_step))
#ax01.set_yticks(np.arange(Y_min, Y_max+Y_step, Y_step))
ax01.tick_params(axis='both', labelsize=12)
for label in ax01.get_xticklabels() + ax01.get_yticklabels():
    label.set_fontweight('bold')
ax01.set_title(ax01_saveTitle, fontweight=labelFontWeight)
ax01.grid(True)

###----------Figure Finalization
#fig1.tight_layout()
plt.savefig(saveTitle, dpi=dpi_res)
plt.show()
#plt.close()

#sys.exit()




##########=========================     PLOTTING Bmag(r,z)     =========================##########
figTitle = 'Magnetic Field Magnitude'
saveTitle = str(re.sub(r'[\\/*?:"<>|\n$]', '_', figTitle))
ax0_saveTitle = saveTitle
ax1_saveTitle = 'Magnetic Field Midplane Lineout'

cmapName = 'Spectral_r'
cbarLabel = r"$\boldsymbol{|B(r,z)|}$  $\boldsymbol{[T]}$"
labelFontWeight = 'bold'
lineWidth = 2

Lconv = 1e-3
X = R/Lconv          # r in mm (cell centers)
Y = Z/Lconv          # z in mm
vmin = 0
vmax = axesLimit(Bmag,1)



###----------Color bar management for dependent variable
USE_LOG = False     #True -> LogNorm, False -> linear

#copy cmap and set the "under"/"over" colors to be exactly the end colors
cmap = plt.colormaps[cmapName].copy()
cmap.set_under(cmap(0))          # below vmin -> same as lowest color
cmap.set_over(cmap(cmap.N - 1))  # above vmax -> same as highest color

numLvlCol = 100     #greatly affects processing speed but you do get a sexy color bar; maybe save 1000 for final plots
if USE_LOG:
    norm   = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.geomspace(vmin, vmax, numLvlCol)
    Zplot  = np.where(Bmag > 0, Bmag, np.nan)  # LogNorm needs > 0
    Zplot = np.where(Bmag <= 0.0, vmin/10.0, Bmag)   # <-- key line
else:
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    levels = np.linspace(vmin, vmax, numLvlCol)
    Zplot  = Bmag



###----------Ax1: Magnetic Field Contour
fig1 = plt.figure(figsize=(figXsize, figYsize))
ax0 = fig1.add_axes([Xax1_start, Yax1_start, Xax1_len, Yax1_len])     #[x, y, width, height]
ax1 = fig1.add_axes([Xax2_start, Yax2_start, Xax2_len, Yax2_len])     
contour = ax0.contourf(
    X, Y, Zplot, levels=levels, cmap=cmap, norm=norm, extend='both'
)
cbar = fig1.colorbar(contour, ax=ax0, extend='both', label=cbarLabel)
if USE_LOG:
    ticks = np.geomspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([r"$\boldsymbol{" + f"{t:.1f}" + r"}$" for t in ticks])
else:
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([r"$\boldsymbol{" + f"{t:.1f}" + r"}$" for t in ticks])
    

ax0.set_xlabel(r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$")
ax0.set_ylabel(r"$\boldsymbol{z}$  $\boldsymbol{[mm]}$")
ax0.tick_params(axis='both', labelsize=12)
for label in ax0.get_xticklabels() + ax0.get_yticklabels():
    label.set_fontweight('bold')
ax0.set_xlim(-Rd/Lconv, Rd/Lconv)
ax0.set_ylim(-Zd/Lconv, Zd/Lconv)
ax0.set_title(ax0_saveTitle, fontweight=labelFontWeight)
ax0.set_aspect('equal', adjustable='box')



###----------Ax2: Magnetic Field Midplane Lineout
X_array = rSlice/Lconv
Y_array = BSlice


X_min   = 0
X_max   = np.ceil(X_array[-1])
X_step  = 10**np.floor(np.log10(X_array[-1]))
Y_min   = axesLimit(Y_array, -1)
Y_max   = axesLimit(Y_array, 1)
Y_step  = 10

X_label = r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$"
Y_label = r"$\boldsymbol{B(r,0)}$  $\boldsymbol{[T]}$"

ax1.plot(X_array, Y_array, linewidth=lineWidth)
ax1.set_xlabel(X_label)
ax1.set_ylabel(Y_label)
ax1.set_xlim(X_min, X_max)
ax1.set_ylim(Y_min, Y_max)
ax1.set_xticks(np.arange(X_min, X_max+X_step, X_step))
ax1.set_yticks(np.arange(Y_min, Y_max+Y_step, Y_step))
ax1.tick_params(axis='both', labelsize=12)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight('bold')
ax1.set_title(ax1_saveTitle, fontweight=labelFontWeight)
ax1.grid(True)

###----------Figure Finalization
#fig1.tight_layout()
plt.savefig(saveTitle, dpi=dpi_res)
plt.show()
#plt.close()






#sys.exit()
###------------------------------------------------------------------------------------------------DIV(B)
dBr__dr, dBr__dz = np.gradient(Br, r, z, edge_order=2)  #gradient of Br [T/m]
dBz__dr, dBz__dz = np.gradient(Bz, r, z, edge_order=2)  #gradient of Bz [T/m]

#add in my functions here and compare to see new modB graphs

divB = dBr__dr + Br/R + dBz__dz                         #divergence of B [T/m]
divB_scaled = np.abs(divB) / (Bw/Rs)                    #scaled divergence of B [\]



###-----Debugging for the numerical anomalies in the divB calcs
idx = 0
idy = idx
#print("Divergence of B:")
#print("\tdBr__dr \t Br/R \t\t dBz__dz \t divB \t\t divB_scaled")
#print(
#    f"\t{dBr__dr[idx,idy]} "
#    f"{Br[idx,idy]/R[idx,idy]} "
#    f"{dBz__dz[idx,idy]} "
#    f"{divB[idx,idy]} "
#    f"{divB_scaled[idx,idy]}"
#)
#print("\t divB_scaled max and min", divB_scaled.max(), divB_scaled.min())


















###-----------------------------------------------------------------------------------CURRENT DENSITY
J = (1 / mu0) * (dBr__dz - dBz__dr)     #current density [A/m^2]
Jphi = -J


JSlice = Jphi[mask, mid_z]    # shape (Nr_pos,)

##########=========================     PLOTTING J(r,z)     =========================##########
figTitle = 'Current Density Magnitude'
saveTitle = str(re.sub(r'[\\/*?:"<>|\n$]', '_', figTitle))
ax2_saveTitle = saveTitle
ax3_saveTitle = 'Current Density Midplane Lineout'

cmapName = 'plasma'
#cbarLabel = r"$\boldsymbol{-J_{\phi}(r,z)}$  $\boldsymbol{[A/m^2]}$"
cbarLabel = r"$\boldsymbol{-J_{\phi}}$  $\boldsymbol{[A/m^2]}$"
labelFontWeight = 'bold'
lineWidth = 2

Lconv = 1e-3
X = R/Lconv          # r in mm (cell centers)
Y = Z/Lconv          # z in mm
vmin = np.abs(Jphi).min()
vmax = np.abs(Jphi).max()



###----------Color bar management for dependent variable
USE_LOG = True     #True -> LogNorm, False -> linear

#copy cmap and set the "under"/"over" colors to be exactly the end colors
cmap = plt.colormaps[cmapName].copy()
cmap.set_under(cmap(0))          # below vmin -> same as lowest color
cmap.set_over(cmap(cmap.N - 1))  # above vmax -> same as highest color

if USE_LOG:
    norm   = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.geomspace(vmin, vmax, 100)
    Zplot  = np.where(Jphi > 0, Jphi, np.nan)  # LogNorm needs > 0
    Zplot = np.where(Jphi <= 0.0, vmin/10.0, Jphi)   # <-- key line
else:
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    levels = np.linspace(vmin, vmax, 100)
    Zplot  = Jphi



###----------Ax2: Current Density Contour
fig2 = plt.figure(figsize=(figXsize, figYsize))
ax2 = fig2.add_axes([Xax1_start, Yax1_start, Xax1_len, Yax1_len])     #[x, y, width, height]
ax3 = fig2.add_axes([Xax2_start, Yax2_start, Xax2_len, Yax2_len])     
contour = ax2.contourf(
    X, Y, Zplot, levels=levels, cmap=cmap, norm=norm, extend='both'
)
cbar = fig2.colorbar(contour, ax=ax2, extend='both', label=cbarLabel)
if USE_LOG:
    ticks = np.geomspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([cbar_label(t) for t in ticks])
else:
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([cbar_label(t) for t in ticks])

ax2.set_xlabel(r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$")
ax2.set_ylabel(r"$\boldsymbol{z}$  $\boldsymbol{[mm]}$")
ax2.tick_params(axis='both', labelsize=12)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight('bold')
ax2.set_xlim(-Rd/Lconv, Rd/Lconv)
ax2.set_ylim(-Zd/Lconv, Zd/Lconv)
ax2.set_title(ax2_saveTitle, fontweight=labelFontWeight)
ax2.set_aspect('equal', adjustable='box')





###----------Ax3: Magnetic Field Midplane Lineout
X_array = rSlice/Lconv
Y_array = JSlice


X_min   = 0
X_max   = np.ceil(X_array[-1])
X_step  = 10**np.floor(np.log10(X_array[-1]))
#Y_min   = axesLimit(Y_array, -1)
#Y_max   = axesLimit(Y_array, 1)
#Y_step  = 10

X_label = r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$"
Y_label = r"$\boldsymbol{-J_{\phi}(r,0)}$  $\boldsymbol{[A/m^{2}]}$"

ax3.plot(X_array, Y_array, linewidth=lineWidth)
ax3.set_xlabel(X_label)
ax3.set_ylabel(Y_label)
ax3.set_xlim(X_min, X_max)
#ax3.set_ylim(Y_min, Y_max)
ax3.set_xticks(np.arange(X_min, X_max+X_step, X_step))
#ax3.set_yticks(np.arange(Y_min, Y_max+Y_step, Y_step))
ax3.tick_params(axis='both', labelsize=12)
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontweight('bold')
ax3.set_title(ax3_saveTitle, fontweight=labelFontWeight)
ax3.grid(True)

###----------Figure Finalization
#fig3.tight_layout()
plt.savefig(saveTitle, dpi=dpi_res)
plt.show()
#plt.close()




###------------------------------------------------------------------------------------------PRESSURE
dP__dr = Jphi * Bz          #partial of pressure wrt radius [Pa/m]
dP__dz = - Jphi * Br        #partial of pressure wrt z [Pa/m]

if model=="sporer":
    P_int = pressure_sporer(Bw, Xs, psi_int, a, b, f)       #his Be is my (and stein's) Bw
    P = np.where(inside, P_int, 0)
elif model=="stein":
    #P = pressure_stein(r, z, Jphi, Br, Bz)
    P_int = pressure_jeff_mesh(R, Z, -Jphi, Br, Bz)
    P = np.where(inside, P_int, 0)

Pscaled = P/(Bw**2/(2*mu0))
#print(Bw, Xs, psix, a, b, f, mu0)
#Px = pressure_sporer(Bw, Xs, psix, a, b, f)       #his Be is my (and stein's) Bw
#nx = Px / (kb * T)
#print("Px = ", Px, "\t\t nx = ", nx)

###-----Debugging - Error Check
# Compute numerical partials of your reconstructed P(r,z):
dPdr_check, dPdz_check = np.gradient(P, r, z, edge_order=2)

# Compare to the original dP__dr, dP__dz:
#print("Pressure:")
#print("\tMax Pressure = {:.3e} [Pa]".format(P_int.argmax()))
#print("\t Max |dPdr_check - dP__dr| =", np.nanmax(np.abs(dPdr_check - dP__dr)))
#print("\t Max |dPdz_check - dP__dz| =", np.nanmax(np.abs(dPdz_check - dP__dz)))

PSlice = P[mask, mid_z]    # shape (Nr_pos,)



##########=========================     PLOTTING P(r,z)     =========================##########
figTitle = 'Plasma Pressure Magnitude'
saveTitle = str(re.sub(r'[\\/*?:"<>|\n$]', '_', figTitle))
ax4_saveTitle = saveTitle
ax5_saveTitle = 'Plasma Pressure Midplane Lineout'

cmapName = 'turbo_r'
cbarLabel = r"$\boldsymbol{P(r,z)}$  $\boldsymbol{[Pa]}$"
labelFontWeight = 'bold'
lineWidth = 2

Lconv = 1e-3
X = R/Lconv          # r in mm (cell centers)
Y = Z/Lconv          # z in mm
Zvar = P

vmin = 0
vmax = np.abs(Zvar).max()



###----------Color bar management for dependent variable
USE_LOG = False     #True -> LogNorm, False -> linear

#copy cmap and set the "under"/"over" colors to be exactly the end colors
cmap = plt.colormaps[cmapName].copy()
cmap.set_under(cmap(0))          # below vmin -> same as lowest color
cmap.set_over(cmap(cmap.N - 1))  # above vmax -> same as highest color

if USE_LOG:
    norm   = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.geomspace(vmin, vmax, 100)
    Zplot  = np.where(Zvar > 0, Zvar, np.nan)  # LogNorm needs > 0
    Zplot = np.where(Zvar <= 0.0, vmin/10.0, Zvar)   # <-- key line
else:
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    levels = np.linspace(vmin, vmax, 100)
    Zplot  = Zvar



###----------Ax4: Plasma Pressure Contour
fig3 = plt.figure(figsize=(figXsize, figYsize))
ax4 = fig3.add_axes([Xax1_start, Yax1_start, Xax1_len, Yax1_len])     #[x, y, width, height]
ax5 = fig3.add_axes([Xax2_start, Yax2_start, Xax2_len, Yax2_len])    
contour = ax4.contourf(
    X, Y, Zplot, levels=levels, cmap=cmap, norm=norm, extend='both'
)
cbar = fig3.colorbar(contour, ax=ax4, extend='both', label=cbarLabel)
if USE_LOG:
    ticks = np.geomspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([cbar_label(t) for t in ticks])
else:
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([cbar_label(t) for t in ticks])
    

ax4.set_xlabel(r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$")
ax4.set_ylabel(r"$\boldsymbol{z}$  $\boldsymbol{[mm]}$")
ax4.tick_params(axis='both', labelsize=12)
for label in ax4.get_xticklabels() + ax4.get_yticklabels():
    label.set_fontweight('bold')
ax4.set_xlim(-Rd/Lconv, Rd/Lconv)
ax4.set_ylim(-Zd/Lconv, Zd/Lconv)
ax4.set_title(ax4_saveTitle, fontweight=labelFontWeight)
ax4.set_aspect('equal', adjustable='box')



###----------Ax5: Magnetic Field Midplane Lineout
X_array = rSlice/Lconv
Y_array = PSlice


X_min   = 0
X_max   = np.ceil(X_array[-1])
X_step  = 10**np.floor(np.log10(X_array[-1]))
#Y_min   = axesLimit(Y_array, -1)
#Y_max   = axesLimit(Y_array, 1)
#Y_step  = 1e9

X_label = r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$"
Y_label = r"$\boldsymbol{P(r,0)}$  $\boldsymbol{[Pa]}$"

ax5.plot(X_array, Y_array, linewidth=lineWidth)
ax5.set_xlabel(X_label)
ax5.set_ylabel(Y_label)
ax5.set_xlim(X_min, X_max)
#ax5.set_ylim(Y_min, Y_max)
ax5.set_xticks(np.arange(X_min, X_max+X_step, X_step))
#ax5.set_yticks(np.arange(Y_min, Y_max+Y_step, Y_step))
ax5.tick_params(axis='both', labelsize=12)
for label in ax5.get_xticklabels() + ax5.get_yticklabels():
    label.set_fontweight('bold')
ax5.set_title(ax1_saveTitle, fontweight=labelFontWeight)
ax5.grid(True)

###----------Figure Finalization
#fig3.tight_layout()
plt.savefig(saveTitle, dpi=dpi_res)
plt.show()
#plt.close()






###------------------------------------------------------------------------------------NUMBER DENSITY

n_edges = P / (kb * TK)

r0_idx = np.searchsorted(r, 0.0)        # first r >= 0
z0_idx = np.abs(z - 0.0).argmin()       # column nearest z=0

start = r0_idx // 2
n_midplane = n_edges[start:, z0_idx]


#Compute cell-centered number density by averaging 2x2 blocks
n_cc = 0.25 * (
    n_edges[:-1, :-1] +  # top-left
    n_edges[1:, :-1] +   # bottom-left
    n_edges[:-1, 1:] +   # top-right
    n_edges[1:, 1:]      # bottom-right
)


ixe = 1                         #ratio of number of ions to electrons (1=quasineutrality)
ni_cc = n_cc / (1 + 1/ixe)      #cell-centered ion density assuming quasineutrality [m^-3]
ne_cc = n_cc / (1 + ixe)        #cell-centered electron density assuming quasineutrality [m^-3]



#####---NUMBER DENSITY DEBUG
n_max_edges = n_edges.max()
n_max_cc = n_cc.max()
ni_max = ni_cc.max()
ne_max = ne_cc.max()


r_cc = 0.5*(r[:-1] + r[1:])
z_cc = 0.5*(z[:-1] + z[1:])

r0_idx = np.searchsorted(r_cc, 0.0)
z0_idx = np.abs(z_cc - 0.0).argmin()

r_mid   = r_cc[r0_idx:]                    # r ≥ 0 
n_mid  = n_cc[r0_idx:, z0_idx]                   # midplane number density
ni_mid  = ni_cc[r0_idx:, z0_idx]                   # midplane ion number density
ne_mid  = ne_cc[r0_idx:, z0_idx]                   # midplane electron number density
ni_mid = n_mid / 2
ne_mid = n_mid / 2
rho_mid = m_ave * ni_mid + m_e * ne_mid                          # mass density [kg m^-3] if m_ave in kg

rSlice = r_mid
nSlice = n_mid
rhoSlice = rho_mid

#average densities
ni_ave = np.mean(ni_mid)
ni_ave_cc = np.mean(ni_cc)

ni_ave = ni_mid[ni_mid != 0].mean()
ni_ave_cc = ni_cc[ni_cc != 0].mean()


rho_cc = ni_cc * m_ave          #+ ne_cc * m_e       #mass density [kg/m^3]
rho_cc = rho_cc * (1e-3)    #mass density [g/cm^3]



##########=========================     PLOTTING n(r,z)     =========================##########
figTitle = 'Cell-Centered Density'
saveTitle = str(re.sub(r'[\\/*?:"<>|\n$]', '_', figTitle))
ax4_saveTitle = saveTitle
ax5_saveTitle = 'Cell-Centered Midplane Density (r > 0)'

cmapName = 'turbo'
#cbarLabel = r"$\boldsymbo{n(r,z)}$  $[m^{-3}]$"
cbarLabel = r"$\boldsymbol{n(r,z)}$  $\boldsymbol{[m^{-3}]}$"
labelFontWeight = 'bold'
lineWidth = 2

Lconv = 1e-3
X = Rcc/Lconv          # r in mm (cell centers)
Y = Zcc/Lconv          # z in mm
Zvar = rho_cc

vmin = 1e-5
vmax = 1e-1


###----------Color bar management for dependent variable
USE_LOG = True     #True -> LogNorm, False -> linear

#copy cmap and set the "under"/"over" colors to be exactly the end colors
cmap = plt.colormaps[cmapName].copy()
cmap.set_under(cmap(0))          # below vmin -> same as lowest color
cmap.set_over(cmap(cmap.N - 1))  # above vmax -> same as highest color

if USE_LOG:
    norm   = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.geomspace(vmin, vmax, 100)
    Zplot  = np.where(Zvar > 0, Zvar, np.nan)  # LogNorm needs > 0
    Zplot = np.where(Zvar <= 0.0, vmin/10.0, Zvar)   # <-- key line
else:
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    levels = np.linspace(vmin, vmax, 100)
    Zplot  = Zvar


###----------Ax6: Density Contour
fig4 = plt.figure(figsize=(figXsize, figYsize))
ax6 = fig4.add_axes([Xax1_start, Yax1_start, Xax1_len, Yax1_len])     #[x, y, width, height]
ax7 = fig4.add_axes([Xax2_start, Yax2_start, Xax2_len, Yax2_len])     
contour = ax6.contourf(
    X, Y, Zplot, levels=levels, cmap=cmap, norm=norm, extend='both'
)
cbar = fig4.colorbar(contour, ax=ax6, extend='both', label=cbarLabel)
if USE_LOG:
    ticks = np.geomspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([cbar_label(t) for t in ticks])
else:
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([cbar_label(t) for t in ticks])    

ax6.set_xlabel(r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$")
ax6.set_ylabel(r"$\boldsymbol{z}$  $\boldsymbol{[mm]}$")
ax6.tick_params(axis='both', labelsize=12)
for label in ax6.get_xticklabels() + ax6.get_yticklabels():
    label.set_fontweight('bold')
ax6.set_xlim(-Rd/Lconv, Rd/Lconv)
ax6.set_ylim(-Zd/Lconv, Zd/Lconv)
ax6.set_title(ax4_saveTitle, fontweight=labelFontWeight)
ax6.set_aspect('equal', adjustable='box')



###----------Ax7: Density Midplane Lineout
X_array = rSlice/Lconv
Y_array = nSlice


X_min   = 0
X_max   = np.ceil(X_array[-1])
X_step  = 10**np.floor(np.log10(X_array[-1]))
#Y_min   = axesLimit(Y_array, -1)
#Y_max   = axesLimit(Y_array, 1)
#Y_step  = 1e9

X_label = r"$\boldsymbol{r}$  $\boldsymbol{[mm]}$"
Y_label = r"$\boldsymbol{n(r,0)}$  $\boldsymbol{[m^{-3}]}$"

ax7.plot(X_array, Y_array, linewidth=lineWidth)
ax7.set_xlabel(X_label)
ax7.set_ylabel(Y_label)
ax7.set_xlim(X_min, X_max)
#ax7.set_ylim(Y_min, Y_max)
ax7.set_xticks(np.arange(X_min, X_max+X_step, X_step))
#ax7.set_yticks(np.arange(Y_min, Y_max+Y_step, Y_step))
ax7.tick_params(axis='both', labelsize=12)
for label in ax7.get_xticklabels() + ax7.get_yticklabels():
    label.set_fontweight('bold')
ax7.set_title(ax1_saveTitle, fontweight=labelFontWeight)
ax7.grid(True)





###-------------------------------------------------------------------------LEFT-RIGHT AXIS

###-------------------------------------------------------------
### FIGURE: Dual Y-Axis Plot
###-------------------------------------------------------------
labelFontSize = 12
lineWidth = 3
figXsize = 10
figYsize = 6

X_array = rSlice/Lconv
Y1_array = nSlice
Y2_array = rhoSlice * (1e-3)

X_min, X_max, X_nticks = 0, 6, 7
Y1_min, Y1_max, Y1_nticks = 0, np.max(n_mid)*(1.2), 11
Y2_min, Y2_max, Y2_nticks = 0, np.max(rho_mid)*(1e-3)*(1.2), 11

X_label  = r"$\boldsymbol{r}$ $\boldsymbol{[mm]}$"
Y1_label = r"Total Number Density, $\boldsymbol{n_{cc}(r,0)}$  $\boldsymbol{[m^{-3}]}$"
Y2_label = r"Mass Density, $\boldsymbol{\rho_{cc}(r,0)}$  $\boldsymbol{[gcc]}$"



ax8 = ax7.twinx()
ax7.plot(
    X_array,
    Y1_array,
    linewidth=lineWidth,
    linestyle='-',
    color='blue',
    label=Y1_label
)

ax8.plot(
    X_array,
    Y2_array,
    linewidth=lineWidth,
    linestyle='--',
    color='red',
    label=Y2_label
)

# --- labels ---
ax7.set_xlabel(X_label, fontsize=labelFontSize, fontweight=fontWeight)
ax7.set_ylabel(Y1_label, fontsize=labelFontSize, fontweight=fontWeight, color='blue')
ax8.set_ylabel(Y2_label, fontsize=labelFontSize, fontweight=fontWeight, color='red')

ax7.set_title(figTitle, fontsize=labelFontSize, fontweight=fontWeight)

# --- limits ---
ax7.set_xlim(X_min, X_max)
ax7.set_ylim(Y1_min, Y1_max)
ax8.set_ylim(Y2_min, Y2_max)

# --- ticks ---
ax7.set_xticks(np.linspace(X_min, X_max, X_nticks))
ax7.set_yticks(np.linspace(Y1_min, Y1_max, Y1_nticks))
ax7.tick_params(axis='y', labelcolor='blue')
ax8.tick_params(axis='y', labelcolor='red')
ax8.set_yticks(np.linspace(Y2_min, Y2_max, Y2_nticks))

ax7.tick_params(axis="both", which="major", labelsize=tickFontSize, colors='blue')
ax8.tick_params(axis="y", which="major", labelsize=tickFontSize, colors='red')
for label in ax8.get_yticklabels():
    label.set_fontweight('bold')
ax8.yaxis.set_major_formatter(FuncFormatter(sci_no_pad))

# --- grid ---
ax7.grid(True)


###----------Figure Finalization
#fig4.tight_layout()
plt.savefig(saveTitle, dpi=dpi_res)
plt.show()
#plt.close()


###---------------------------------------------------------------------------------------------OUTPUT
summary_text = frc00.summary_text

summary_text += (
    f"FRC_04: EQUILIBRIUM FORMATION  ============================================= \n"
    f"\t <B_i> = {ave_Bi:.3f} [T] \n"
    f"\t Bmax = {Bmax:.3f} [T] \n"
)

if model in ('stein', 'sporer'):
    summary_text += (
        f"STEINHAUER PARAMETERS ------------------------------------------\n"
        f"\t sig = {sig:.2f} (Adjustable Parameter)\n"
        f"\n"
        f"\t N  = {N:.4f} (Shape Index; 0=racetrack 1=ellipse)\n"
        f"\t E0 = {E0:.4f}\n"
        f"\t E1 = {E1:.4f}\n"
        f"\t E2 = {E2:.4f}\n"
        f"\t alpha = {alpha:.4f} (0 < alpha < 1)\n"
        f"\n"
        f"\t Be+ = {Beplus:.3f} [T] \n"
        f"\t b0 = {b0:.4f} [T]\n"
        f"\t b1 = {b1:.4f} [T]\n"
        f"\t D0 = {D0:.4f}\n"
        f"\t D1 = {D1:.4f}\n"
        f"\n"
    )


summary_text += (
    f"NUMBER DENSITY ------------------------------------------------\n"
    f"n_max = {n_mid.max():.3e} [m^-3]\n"
    f"\t Max Number Density (n_edges.argmax) = {n_max_edges:.3e} [m^-3]\n"
    f"\t Max Number Density (n_cc.argmax) = {n_max_cc:.3e} [m^-3]\n"
    f"\t Max Ion Density (n_i,max) = {ni_max:.3e} [m^-3]\n"
    f"\t Max Ion Density (ni_mid,max) = {ni_mid.max():.3e} [m^-3]\n"
    f"\t Max Electron Density (n_e,max) = {ne_max:.3e} [m^-3]\n"
    f"\t Max Electron Density (ne_mid.max) = {ne_mid.max():.3e} [m^-3]\n"
    f"\t Max Mass Density (rho_cc.max) = {rho_cc.max():.3e} [g/cm^3]\n"
    f"\t Max Mass Density (rho_mid.max) = {rho_mid.max():.3e} [kg/m^3]\n"
    f"\t Average Midplane Ion Number Density (n_i,ave) = {ni_ave:.3e} [m^-3]\n"
    f"\t Average Overall Ion Number Density (n_i,ave) = {ni_ave_cc:.3e} [m^-3]\n"
 )


print(summary_text)
with open("adventure_log.txt", "w") as alog:
    alog.write(summary_text)
sys.exit()









































