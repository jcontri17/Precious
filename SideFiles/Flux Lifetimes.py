#######################################################################################################
#######################################################################################################
#################################                  IMPORTS                 ############################
#######################################################################################################
#######################################################################################################
import matplotlib.pyplot as plt #library for plotting 
import numpy as np              #library for basic math functions 
import cmath                    #library for complex numbers
import sys                      #library for various system-specific parameters and functions
import time                     #library for time-based functions

from plottingParameters import *  
from functions4plasma import*

from scipy.optimize import fsolve
from scipy.optimize import root 
from mpl_toolkits.mplot3d import Axes3D                         #registers the 3D projection 
from matplotlib.ticker import AutoMinorLocator
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator 
from matplotlib.colors import LogNorm
from scipy.integrate import cumulative_trapezoid as cumtrapz


#######################################################################################################
#######################################################################################################
#################################                  MAIN                    ############################
#######################################################################################################
#######################################################################################################

Lconv = 1e-2
model = "sporer"
###---------------------------------------------------------------------------------CONSTANT PARAMETERS 
Rw = 0.61*Lconv         #wall radius [m]
Rc = Rw                 #coil radius [m]

sig = 1.5               #flare parameter; adjustable parameter that's fixed for Steinhauer's paper
f = 1.5                 #internal psi error factor for Sporer's approximation

Xs = 0.75               #normalized separatrix ratio [\]
Rs = Xs * Rw            #separatrix radius [m]
Lx = 3.00*Lconv        #FRC length [m]
Z0 = Lx / 2             #FRC half-length [m]
B0 = 30                 #applied field B0 from Sporer's unreleased paper 
Bw = 10                 #Magnetic field at the midplane at the wall [T] from Steinhaur
B00 = Bw                #Sporer vacuum field [T] still working out how this relates to Steinhauer
a = Rs                  #FRC semi-minor axis [m]
b = Z0                  #FRC semi-major axis [m]
eps = a / b             #inverse elongation

e = 1.6 *10**(-19)      #Fundamental Charge 
eV_J = 6.2415*10**(18)  #amount of eV in a Joule 
eV_K = 11604.525        #the amount of kelvin in eV
k_BB = 1.3806*10**(-23)  #Boltzmann constant (in SI) 
k_B = k_BB * eV_K       #Boltzmann constant [J/eV]
m_deuteron = 3.343583776*10**(-27) - m_e #mass of deuterium  

Nu_normal = 1.03* 10**(-4)            #classical cross-field Spitzer resistivity (N_normal_clas) from Sporer's paper about flux lifetimes 
A_brems = 1.6*10**(-38)     #from Sporer's paper about flux lifetimes [Wm^3 / sqrt(eV)]


acc = 100               #number of variables in list -- accuracy 

###----------------------------------------------------------------------------------------DOMAIN SETUP
h = 4.00*Lconv         #length of liner [m]
domx = 1.1                                  #weight to extend the domain; helps show psi=1 curves better
Rd = Rw * domx                              #half-length of computational domain in r-dir [m]
Zd = h/2 * domx                             #half-length of computational domain in z-dir [m]

dr = 0.005*Lconv                           #mesh fidelity in r-dir [m]
dz = 0.005*Lconv                           #mesh fidelity in z-dir [m]


r = np.arange(-Rd, Rd+dr, dr)               #r-array for mesh
z = np.arange(-Zd, Zd+dz, dz)               #z-array for mesh
R, Z = np.meshgrid(r, z, indexing='ij')     #(R,Z) mesh elements

###---------------------------------------------------------------------------------FUNCTIONS

def pressure_jeff_mesh(R, Z, Jphi, Br, Bz):
    """
    Compute pressure P(r,z) on a 2D (R,Z) mesh using Jeff’s method:

      ∂P/∂r = Jphi * Bz
      ∂P/∂z = -Jphi * Br    (not used for integration here)

    We enforce P(r_max, z) = 0 at the outermost r, and integrate inward
    along each fixed‐z column.  

    Inputs:
      - R    : 2D array, shape (Nr, Nz), radial mesh (meters).  Must satisfy R[i,j] = r_i.
      - Z    : 2D array, shape (Nr, Nz), axial  mesh (meters).  Must satisfy Z[i,j] = z_j.
      - Jphi : 2D array, shape (Nr, Nz), J_φ at each (r_i, z_j) [A/m²].
      - Br   : 2D array, shape (Nr, Nz), B_r(r_i, z_j) [T].
      - Bz   : 2D array, shape (Nr, Nz), B_z(r_i, z_j) [T].

    Returns:
      - P : 2D array, shape (Nr, Nz), pressure P(r_i, z_j) [Pa], with P(r_max, z_j)=0.
    """

    Nr, Nz = R.shape

    # 1) Extract the 1D r‐axis from the first column of R:
    #    We assume R was created via meshgrid(r, z, indexing='ij'),
    #    so R[i,j] == r[i].  Thus r = R[:,0].
    r_1d = R[:, 0].copy()       # shape = (Nr,)

    # 2) Compute ∂P/∂r = Jphi * Bz for every mesh point:
    dP__dr = Jphi * Bz          # shape = (Nr, Nz)

    # (We also could form dP__dz = -Jphi * Br, but it's not needed for the radial integration.)

    # 3) Allocate the output pressure array, initially zeros:
    P = np.zeros_like(Jphi)     # shape = (Nr, Nz)

    # 4) Build a reversed‐r array so that index‐0 corresponds to r_max:
    r_rev = r_1d[::-1]          # shape = (Nr,) from largest r to smallest

    # 5) Loop over each z‐column (fixed j).  Integrate dP/dr from r_max → r=0:
    for j in range(Nz):
        # 5a) Extract column j of dP/dr, then reverse it in r:
        col_dPdr     = dP__dr[:, j]     # shape = (Nr,)
        col_dPdr_rev = col_dPdr[::-1]    # now r_rev[0] is the maximum r

        # 5b) Cumulative trapezoidal integration along reversed‐r axis.
        #     cumtrapz(y, x, initial=0) returns an array Y of length len(x),
        #     where Y[k] = ∫_{x[0]}^{x[k]} y(x') dx'.  Here x[0] = r_rev[0] = r_max.
        integral_rev = cumtrapz(col_dPdr_rev, r_rev, initial=0.0)
        # By construction, integral_rev[0] = 0 at r_rev[0]=r_max, so P(r_max)=0.

        # 5c) Flip back to original r‐ordering and negate to get P(r_i):
        #     integral_rev[::-1][i] = ∫_{r_i}^{r_max} dP/dr' dr'.
        #     We want P(r_i) = - ∫_{r_i}^{r_max} (dP/dr') dr'  (so that ∂P/∂r = dP/dr).
        P[:, j] = - integral_rev[::-1]

    return P

def pressure_sporer(Be, Xs, psi_int, a, b, f):
    eps = a / b
    Pe = Be**2 / (2*mu0)
    x = 3/2 * Xs**2
    y = np.sqrt(3/2) * Xs * 8 * psi_int / (Be*a**2)
    z = f * eps**2

    P = Pe * (1 - x + y + z)
    return P

def sporer_Be(Xs, B0):
    Be = B0 / (1 - Xs**2)
    return Be

### ================================== EXTERNAL FUNCTIONS 

def external_dpsi__dr(r, z, Bw, a, b, E0, E1, E2, alpha):
    denom1 = (alpha*b-z)**2 + r**2
    denom2 = (alpha*b+z)**2 + r**2
    T0 = E0 * r
    T1 = E1 * (r/a**2) * (2*r**2 - 4*z**2)
    T2 = E2 * (a**2/2) * (r*(alpha*b-z)/(denom1**1.5) + r*(alpha*b+z)/(denom2**1.5))

    dpsi__dr = Bw * (T0 + T1 - T2)
    return dpsi__dr

def external_dpsi__dz(r, z, Bw, a, b, E0, E1, E2, alpha):
    denom1 = (alpha*b-z)**2 + r**2
    denom2 = (alpha*b+z)**2 + r**2
    T1 = E1 * 8*z*r**2/a**4
    T2 = E2 * ((z-alpha*b)**2/(denom1**1.5) - 1/denom1**0.5 + 1/denom2**0.5 - (alpha*b+z)**2/denom2**1.5)
    
    dpsi__dz = Bw * a**2/2 * (T2 - T1)
    return dpsi__dz

def external_Be(Bw, a, b, Xs):
    Be__Bw = 1 + 1.46 * (3.2 + ((b/a) / (1.5-Xs))**4)**(-1)
    Be = Bw * Be__Bw
    return Be

def external_psi(r, z, Bw, a, b, E0, E1, E2, alpha):
    psi_w = Bw * a**2 / 2
    e0 = (r/a)**2
    e1 = (r/a)**2 * ((r/a)**2 - 4*(z/a)**2)
    denom1 = (alpha*b+z)**2 + r**2
    denom2 = (alpha*b-z)**2 + r**2
    e2 = (alpha*b+z)/np.sqrt(denom1) + (alpha*b-z)/np.sqrt(denom2)
    
    psi = psi_w * (E0*e0 + E1*e1 + E2*e2)
    return psi

def external_E_params(Es, eps, Xs, sig):
    E0, E1, E2, a = Es

    eq1 = E0 - E1 * (4/eps**2) + E2 * (2*eps**2*a/(1-a**2)**2)
    eq2 = E0 + E1 + 2*a*E2 / np.sqrt(eps**2+a**2)
    e0 = eps**2*Xs**3*E2/4/sig**2 
    e1 = (a+sig) / np.sqrt(eps**2 + Xs**2*(a+sig)**2)
    e2 = (a-sig) / np.sqrt(eps**2 + Xs**2*(a-sig)**2)
    e3 = 2*a / np.sqrt(eps**2 + (Xs*a)**2)
    eq3 = E1 - e0 * (e1 + e2 - e3)
    eq4 = E0 + 2*E1/Xs**2 - eps**2*a*Xs**3*E2 / ((a*Xs)**2 + eps**2)**(1.5) - 1

    return [eq1, eq2, eq3, eq4]

### ================================== INTERNAL FUNCTIONS 

def internal_psi_sporer(r, z, a, b, Be, Xs, f):     #internal flux
    eps = a / b         #Inverse elongation
    T1 = np.sqrt(3/2)
    T2 = (Xs * Be * r**2) / 2
    T3 = 1 - (r/a)**2 - (z/b)**4 + f * eps**2
    
    psi_int = T1 * T2 * T3 
    return psi_int

def internal_psi_stein(r, z, a, b, B0, B1, D0, D1):
    b0 = 1 - (r/a)**2 - (z/b)**2
    d0 = (r/a)**2 - 4*(z/a)**2
    d1 = (r/a)**4 - 12*(r*z/a/a)**2 + 8*(z/a)**4
    b1 = 1 + D0*d0 + D1*d1

    psi_int = (r**2/2) * (B0*b0 + B1*b1)
    return psi_int

def internal_dpsi__dr_stein(r, z, a, b, B0, B1, D0, D1):
    b0 = 1 - 2*(r/a)**2 - (z/b)**2
    d0 = 2 * (r/a)**2 - 4 * (z/a)**2
    d1 = 3 * (r/a)**4 - 24 * (r*z/a/a)**2 + 8 * (z/a)**4
    b1 = 1 + D0*d0 + D1*d1

    dpsi__dr = (B0 * r * b0) + (B1 * r * b1)
    return dpsi__dr

def internal_dpsi__dz_stein(r, z, a, b, B0, B1, D0, D1):
    b0 = - 2 * z / b**2
    d0 = - 8 * z / a**2
    d1 = -24 * r**2 * z / a**4 + 32 * z**3 / a**4
    b1 = D0 * d0 + D1 * d1
    
    dpsi__dz = (r**2 / 2) * (B0*b0 + B1*b1)
    return dpsi__dz

def internal_dpsi__dr_sporer(r, z, a, b, Be, Xs, f):
    eps = a / b
    F = np.sqrt(3/2) * Xs * Be * r
    P = 1 - 2*(r/a)**2 - (z/b)**4 + f * eps**2
    
    dpsi__dr = F * P
    return dpsi__dr
    
def internal_dpsi__dz_sporer(r, z, a, b, Be, Xs):
    dpsi__dz = - np.sqrt(6) * Xs * Be * r**2 * z**3 / b**4
    return dpsi__dz

def internal_D0(eps):
    D0 = (eps**4-8) / (4*(2+eps**2))
    return D0

def internal_D1(eps):
    D1 = - (eps**2*(4+eps**2)) / (4*(2+eps**2))
    return D1

def internal_B1(B0, D0, D1, N, eps):
    T2 = eps**2 * (N-1)
    T0 = D0 * (4+eps**2*N)
    T1 = D1 * (12 + 2*eps**2 * N)

    B1 = B0 * T2 / (T0 + T1)
    return B1

def internal_B0(a, b, Xs, Bw):
    N = 2.56 - 3.28*Xs
    D = 9.65 - 15.2*Xs + (b/a)**4
    p = 1 + N/(Xs**2 * D)    
    sr = np.sqrt(3/2)*Xs

    B0 = Bw * sr * p
    return B0

### ================================== FLUX LIFETIME FUNCTIONS 
def tau_clas(r_s, T, lamb_A):                   #anomalous flux liftimes for calssical
    Nu = Nu_normal * lamb_A * T**(-3/2)
    result = (1/16) * (mu0*r_s**(2))/(Nu)
    return result 
def tau_LSX(r_s, m, T, B0, r_w):        #anomalous flux lifetimes for LSX
    x_s = r_s/r_w
    p_ie = ((2*m*k_B*T)**(0.5))/(e*B0)  #temperature is in eV*** 
    result = (6.5 * 10**(-5)) * np.sqrt(x_s) * ((r_s)**(2.14)/(p_ie**0.5)**(2.14))
    return result 
def tau_brems(n, T):                    #bremsstrahlung decay time for any hydrogenic plasma
    #n = A*(np.sqrt(2*np.pi*k_B*T)/(np.sqrt(m_deuteron-m_e)))
    result = (2*n*k_B*T)/(A_brems*n**(2)*np.sqrt(k_B*T))*(1/np.sqrt(eV_J))
    return result 
def tau_LSX_num(Rw, n, rs):                #alternative flux lifetime from LSX experiment
    xs = rs/Rw
    result = 0.02 * np.sqrt(xs) * (n**(0.53)) * (rs**(2.14))
    return result
def tau_clas_num(Rw, n, rs):                #alternative flux lifetime from LSX experiment
    xs = rs/Rw
    co = 3/(np.sqrt(xs)*n**(0.53)*rs**(0.14))
    result = (1/16) * (rs**(2))*(1/co)
    return result

### ================================== EXTRANEOUS FUNCTIONS 
def coulomb_log(T, n):      #n is number density
    dummy1 = np.log(np.sqrt(n)*T**(-5/4))
    dummy2 = (10**(-5)*((np.log(T)-2)**2)/16)**(0.5)
    result = 23.5 - dummy1 - dummy2 
    return result

def shape_index_N3(a, b, Xs):           #improved analysis of equilbria --- equation 8 --- Shape Index Function  
    N = (1+(0.8+Xs**2)*(b/a-1)**2)**(-1)
    return N

#######################################################################################################
#######################################################################################################
#################################                 PARAMETERS               ############################
#######################################################################################################
####################################################################################################### 


###-------------------------------------------------------------------------------------EXTERNAL REGION
Eguess = [Bw, Bw/3, Bw/5, 0.9]              #initial guess for E parameters
sol = root(
    fun=external_E_params,
    x0=Eguess,
    args=(eps, Xs, sig),
    method='hybr',   # same algorithm as old fsolve, or try 'lm'
    tol=1e-6
)
if not sol.success:
    print("root() failed to converge:", sol.message)
    Br_ext = Bz_ext = 0
else:
    E0, E1, E2, alpha = sol.x

    psi_ext = external_psi(R, Z, B00, a, b, E0, E1, E2, alpha)
    dpsi__dz_ext = external_dpsi__dz(R, Z, B00, a, b, E0, E1, E2, alpha)   #check these signs
    dpsi__dr_ext = external_dpsi__dr(R, Z, B00, a, b, E0, E1, E2, alpha)
    Br_ext = -(1/R) * dpsi__dz_ext   #check these signs
    Bz_ext = (1/R) * dpsi__dr_ext

###-------------------------------------------------------------------------------------INTERNAL REGION

if model=="stein":
    psi_int = internal_psi_stein(R, Z, a, b, B0, B1, D0, D1)
    Br_int = (1/R) * internal_dpsi__dz_stein(R, Z, a, b, B0, B1, D0, D1)
    Bz_int = -(1/R) * internal_dpsi__dr_stein(R, Z, a, b, B0, B1, D0, D1)
elif model=="sporer":
    psi_int = internal_psi_sporer(R, Z, a, b, Bw, Xs, f)
    Br_int = (1/R) * internal_dpsi__dz_sporer(R, Z, a, b, Bw, Xs)
    Bz_int = -(1/R) * internal_dpsi__dr_sporer(R, Z, a, b, Bw, Xs, f)



###----------------------------------------------------------------MAGNETIC FIELD CONSTRUCTION - B(r,z)
construct = "cutoff"            #net or cutoff

if construct=="cutoff":
    #inside = (psi_int > 0)
    inside = (psi_ext < 0)
    psi = np.where(inside, psi_int, psi_ext)
    Br = np.where(inside, Br_int, Br_ext)
    Bz = np.where(inside, Bz_int, Bz_ext)
elif construct=="net":
    psi = psi_ext - psi_int
    dpsi_dr, dpsi_dz = np.gradient(psi, R, Z, edge_order=2)
    Br = -(1/R) * dpsi_dz
    Bz = (1/R) * dpsi_dr

###--------------------------------------------------------------------------------------CURRENT DENSITY
dBr__dr, dBr__dz = np.gradient(Br, r, z, edge_order=2)
dBz__dr, dBz__dz = np.gradient(Bz, r, z, edge_order=2)
J = (1 / mu0) * (dBr__dz - dBz__dr)     #current density [A/?^2]
Jphi = J

###--------------------------------------------------------------------------------------PRESSURE
if model=="sporer":
    P_int = pressure_sporer(Bw, Xs, psi_int, a, b, f)       #his Be is my (and stein's) Bw
    P = np.where(inside, P_int, 0)
elif model=="stein":
    #P = pressure_stein(r, z, Jphi, Br, Bz)
    P_int = pressure_jeff_mesh(R, Z, -Jphi, Br, Bz)
    P = np.where(inside, P_int, 0)
    
P_avg_inside = np.mean(P[inside])  #shape: (numTemps,)
P_mesh = np.full_like(P, P_avg_inside)

Pscaled = P/(Bw**2/(2*mu0))

###--------------------------------------------------------------------------------------NUMBER DENSITY
Nr, Nz = R.shape

#temperature values (need to change, currently for Sporer's cold and hot FRCs)
temperature_list = np.linspace(100, 500, num=5)  #eV

numTemps = len(temperature_list)
n_array = np.zeros(((Nr, Nz, numTemps)))
n_array_all_scaled = np.zeros(((Nr, Nz, numTemps)))
n_array_each_scaled = np.zeros(((Nr, Nz, numTemps)))
n_array_max= np.zeros(numTemps)

for i in range(numTemps):
    T = temperature_list[i] * eV/kb
    n_array[:,:,i] = P / (kb * T)
    n = P / (kb * T)
    n_array_max[i] = np.max(n_array[:,:,i]) #finding the max for each temperature slice
    #n_max_array[i] = np.max(n_array) --> finding the max number density over entire array

n_array_max_tau = n_array_max* (10**(-21)) #converting to the units of the LSX number density flux equation, specific to only said equation
n_array_max_log = n_array_max * 10 **(-6)   #units from m^-3 to cm^-3

#finding the spatial average 
n_array_avg = np.mean(n_array, axis=(0,1))  #shape: (numTemps,)
    #this computes the mean over both the radial and axial dimensions (Nr, Nz) for each temp
   
n_array_avg_tau = n_array_avg * (10**(-21)) #converting to the units of the LSX number density flux equation, specific to only said equation
n_array_avg_log = n_array_avg * 10 **(-6)   #units from m^-3 to cm^-3


n_min = np.min(n_array)
n_max = np.max(n_array)
levels = np.linspace(n_min, n_max, 100)
levels_scaled = np.linspace(n_min/n_max, n_max/n_max, 100)

print("Number Density:")
print("\t n_max = {:.3e} [m^-3] = {:.3e} [cm^-3]".format(n_max, n_max*Lconv**3))
print("\t n_min = {:.3e} [m^-3] = {:.3e} [cm^-3]".format(n_min, n_min*Lconv**3))

#Coulomb values for corresponding T values
coulomb_list = []
for index, temp in enumerate(temperature_list): 
    coulomb_list.append(coulomb_log(temp, n_array_max_log[index]))
    
###--------------------------------------------------------------------------------------GRAPHING PARAMETERS    
#elongation (b/a) data points between 1 and 10
elongation_list = np.linspace(1,10,num= acc)

#inverse elongation (a/b)
eps_list = 1/elongation_list
b_list = np.linspace(0.0001, 0.0534, num = acc)
a_list = b_list*eps_list                        #a == rs 
Xs_list = a_list / Rw
Xs_listValues = np.linspace(0.3,0.9,num=5)      #X_s values between 0.3 and 0.9 for Shape Index


Shape_Index_N3_list = []                        #Shape Index values consistent with Steinhaurer's paper
for i in Xs_listValues:
    Shape_Index_N3_list.append(shape_index_N3(a_list, b_list, i))
    

#y_lists for tau_clas for every temperature value 
ylist_tau_clas = []
for index, value in enumerate(temperature_list): 
    ylist_tau_clas.append(tau_clas(a_list, value, coulomb_list[index]))


#y_lists for tau_LSX for every temperature value 
ylist_tau_LSX = []
for i in temperature_list: 
    empty_list = []
    for index, value in enumerate(elongation_list):
        empty_list.append(tau_LSX(a_list[index], m_deuteron, i, B0, Rw))
    ylist_tau_LSX.append(empty_list)


#y_list for tau_num & tau_brems 
ylist_tau_LSX_num_avg =[]
ylist_tau_LSX_num_max = []
ylist_tau_clas_num_avg = []
ylist_tau_clas_num_max = []
ylist_tau_brems = []        #bremsstrahlung radiation ylist 
for index, temp in enumerate(temperature_list): 
    ylist_tau_clas_num_max.append(tau_clas_num(Rw, n_array_max_tau[index], a_list))
    ylist_tau_clas_num_avg.append(tau_clas_num(Rw, n_array_avg_tau[index], a_list))
    ylist_tau_LSX_num_avg.append(tau_LSX_num(Rw, n_array_avg_tau[index], a_list))
    ylist_tau_LSX_num_max.append(tau_LSX_num(Rw, n_array_max_tau[index], a_list))
    ylist_tau_brems.append(tau_brems(n_array_max[index], temp))
    
#######################################################################################################
#######################################################################################################
#################################                 PLOTS                    ############################
#######################################################################################################
####################################################################################################### 

plot_num = 0            #indexing for plot #

"""
#overal domain is correct, but the values are still off --> coulombs log is off... 

 
### --- elongation (b/a) is the x-axis, tau_clas is the y-axis
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))

    plt.figure(num = plot_num, dpi = dpi_res)
    plt.plot(elongation_list, ylist_tau_clas[T_index])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{classical}$ vs Elongation, T = ' + temp + "eV", fontsize = titleFontSize)
    plt.xlabel("Elongation [b/a]", fontsize = labelFontSize)            # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    title_save = "tau_clas_T" + temp + " vs elongation"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

### --- Shape Index (N3) is the x-axis, tau_class is the y-axis
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)

    for index, value in enumerate(Shape_Index_N3_list):
        lab_el = "{:.2f}".format(Xs_listValues[index])
        plt.plot(value, ylist_tau_clas[T_index], label = lab_el )

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{classical}$ vs Shape Index, T = ' + temp + "eV", fontsize = titleFontSize)
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)  # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(title = r'$X_s$', fontsize = 'small')
    title_save = "tau_clas_T" + temp + " vs Shape Index"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

#######################################################################################################
####################################################################################################### 


### --- elongation (b/a) is the x-axis, tau_LSX is the y-axis
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.plot(elongation_list, ylist_tau_LSX[T_index])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{LSX}$ vs Elongation, T = ' + temp + "eV", fontsize = titleFontSize)
    plt.xlabel("Elongation (b/a)", fontsize = labelFontSize)  # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    title_save = "tau_LSX_T" + temp + " vs Elongation"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

### --- Shape Index (N3) is the x-axis, tau_LSX is the y-axis

for T_index, T_value in enumerate(temperature_list): 
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi= dpi_res)

    for index, value in enumerate(Shape_Index_N3_list):
        lab_el = "{:.2f}".format(Xs_listValues[index])
        plt.plot(value, ylist_tau_LSX[T_index], label = lab_el )  
        
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{LSX}$ vs Shape Index, T = ' + temp + "eV", fontsize = titleFontSize)
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)  # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(title = r'$X_s$', fontsize = 'small')
    title_save = "tau_LSX_T" + temp + " vs Shape Index"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1

"""

#######################################################################################################
####################################################################################################### 
#################################           COMPARISON PLOTS               ############################
#######################################################################################################
#######################################################################################################

colors = {
    "clas": "C0",        # first color in the cycle
    "LSX": "C1",         # second color
    "avg": "C2",         # third color
    "max": "C3",         # fourth color
}

### --- tau_clas & tau_LSX vs Elongation 
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()

    plt.plot(elongation_list, ylist_tau_clas[T_index], color = colors['clas'], label = r'$\tau^{\Phi}_{clas}$') 
    plt.plot(elongation_list, ylist_tau_LSX[T_index], color = colors['LSX'], label = r'$\tau^{\Phi}_{LSX}$')
    #plt.plot(elongation_list, ylist_tau_LSX_num_avg[T_index], color = colors['avg'], label = r'$\tau^{\Phi}_{Ln-avg}$')
    #plt.plot(elongation_list, ylist_tau_LSX_num_max[T_index], color = colors['max'], label = r'$\tau^{\Phi}_{Ln-max}$')

    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    plt.title(r'$\tau^{\Phi}_{clas}$ & $\tau^{\Phi}_{LSX}$ vs Elongation, T = ' + temp + " eV")
    plt.xlabel("elongation [b/a]", fontsize = labelFontSize)
    plt.ylabel(r'Flux lifetime [s], $\tau$', fontsize = labelFontSize)
    plt.legend(fontsize = 'small')
    title_save = "tau_clas, tau_LSX vs elongation, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

### --- tau_clas vs tau_clas number density vs Elongation
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()

    plt.plot(elongation_list, ylist_tau_clas[T_index], color = colors['clas'], label = r'$\tau^{\Phi}_{clas}$') 
    plt.plot(elongation_list, ylist_tau_clas_num_avg[T_index], color = colors['avg'], label = r'$\tau^{\Phi}_{clas-avg}$')
    plt.plot(elongation_list, ylist_tau_clas_num_max[T_index], color = colors['max'], label = r'$\tau^{\Phi}_{clas-max}$')

    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    plt.title(r'$\tau^{\Phi}_{clas}$, $\tau^{\Phi}_{num-avg}$, $\tau^{\Phi}_{num-max}$ vs Elongation, T = ' + temp + " eV")
    plt.xlabel("elongation [b/a]", fontsize = labelFontSize)
    plt.ylabel(r'Flux lifetime [s], $\tau$', fontsize = labelFontSize)
    plt.legend(fontsize = 'small')
    title_save = "tau_clas, tau_clas_num vs elongation, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 

### --- tau_LSX vs tau_LSX number density vs Elongation
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()

    plt.plot(elongation_list, ylist_tau_LSX[T_index], color = colors['LSX'], label = r'$\tau^{\Phi}_{LSX}$') 
    plt.plot(elongation_list, ylist_tau_LSX_num_avg[T_index], color = colors['avg'], label = r'$\tau^{\Phi}_{LSX-avg}$')
    plt.plot(elongation_list, ylist_tau_LSX_num_max[T_index], color = colors['max'], label = r'$\tau^{\Phi}_{LSX-max}$')

    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    plt.title(r'$\tau^{\Phi}_{LSX}$, $\tau^{\Phi}_{num-avg}$, $\tau^{\Phi}_{num-max}$ vs Elongation, T = ' + temp + " eV")
    plt.xlabel("elongation [b/a]", fontsize = labelFontSize)
    plt.ylabel(r'Flux lifetime [s], $\tau$', fontsize = labelFontSize)
    plt.legend(fontsize = 'small')
    title_save = "tau_LSX, tau_LSX_num vs elongation, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 
    
#######################################################################################################
####################################################################################################### 

### --- tau_clas & tau_LSX vs Shape Index
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()

    for index, value in enumerate(Shape_Index_N3_list): 
        plt.plot(value, ylist_tau_clas[T_index], 
                color=colors["clas"], 
                label = r'$\tau^{\Phi}_{clas}$' if index ==0 else "")
            
    for index, value in enumerate(Shape_Index_N3_list): 
        plt.plot(value, ylist_tau_LSX[T_index], 
                color=colors["LSX"], 
                label = r'$\tau^{\Phi}_{LSX}$' if index ==0 else "")
    
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{clas}$ & $\tau^{\Phi}_{LSX}$ vs Shape Index, T = ' + temp + " eV")
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)                 # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(fontsize = 'small')
    title_save = "tau_clas, tau_LSX vs Shape Index, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 
   
    
### --- tau_LSX & tau_num density (LSX) vs Shape Index
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()
            
    for index, value in enumerate(Shape_Index_N3_list): 
        plt.plot(value, ylist_tau_LSX[T_index], 
                color=colors["LSX"], 
                label = r'$\tau^{\Phi}_{LSX}$' if index ==0 else "")
          
    for index, value in enumerate(Shape_Index_N3_list):
        plt.plot(value, ylist_tau_LSX_num_avg[T_index], 
                color=colors["avg"], 
                label = r'$\tau^{\Phi}_{num-avg}$' if index ==0 else "")
        
    for index, value in enumerate(Shape_Index_N3_list):
        plt.plot(value, ylist_tau_LSX_num_max[T_index], 
                color = colors["max"], 
                label = r'$\tau^{\Phi}_{num-max}$' if index ==0 else "")     
        
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{LSX}$ & $\tau^{\Phi}_{num}$ vs Shape Index, T = ' + temp + " eV")
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)                 # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(fontsize = 'small')
    title_save = "tau_LSX, tau_num vs Shape Index, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 
    
    
### --- tau_clas & tau_num density (clas) vs Shape Index
for T_index, T_value in enumerate(temperature_list):
    temp = str("{:.0f}".format(T_value))
    plt.figure(num = plot_num, dpi = dpi_res)
    plt.grid()
            
    for index, value in enumerate(Shape_Index_N3_list): 
        plt.plot(value, ylist_tau_clas[T_index], 
                color=colors["clas"], 
                label = r'$\tau^{\Phi}_{clas}$' if index ==0 else "")
          
    for index, value in enumerate(Shape_Index_N3_list):
        plt.plot(value, ylist_tau_clas_num_avg[T_index], 
                color=colors["avg"], 
                label = r'$\tau^{\Phi}_{num-avg}$' if index ==0 else "")
        
    for index, value in enumerate(Shape_Index_N3_list):
        plt.plot(value, ylist_tau_clas_num_max[T_index], 
                color = colors["max"], 
                label = r'$\tau^{\Phi}_{num-max}$' if index ==0 else "")     
        
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.title(r'$\tau^{\Phi}_{clas}$ & $\tau^{\Phi}_{num}$ vs Shape Index, T = ' + temp + " eV")
    plt.xlabel("Shape Index [N3]", fontsize = labelFontSize)                 # Label for the x-axis
    plt.ylabel(r'Flux Lifetime [s], $\tau$', fontsize = labelFontSize)     # Label for the y-axis
    plt.legend(fontsize = 'small')
    title_save = "tau_clas, tau_num vs Shape Index, T = " + temp + "eV"
    plt.savefig(title_save + ".png", dpi = dpi_res)
    plot_num += 1 
    
    
    
#print tau_brem value becuase can't graph them 
print(ylist_tau_brems)

#print number density max and min 
print(n_array_max)
print(n_array_avg)

#print coulomb list
print(coulomb_list)