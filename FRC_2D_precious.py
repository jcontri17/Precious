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
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import LogNorm
#from scipy.integrate import cumulative_trapezoid as cumtrapz





t0code = time.perf_counter()
#################################################
#####               FUNCTIONS               #####
#################################################
###----------------------------------------------------------------------------------EXTERNAL FUNCTIONS
###-----EXTERNAL MAGNETIC FLUX FUNCTION, ψ_ext(r,z); Eq(2) in Steinhauer; units = [T*m^2]
def external_psi(r, z, Bw, a, b, E0, E1, E2, alpha):
    psi_w = Bw * a**2 / 2
    e0 = (r/a)**2
    e1 = (r/a)**2 * ((r/a)**2 - 4*(z/a)**2)
    denom1 = (alpha*b+z)**2 + r**2
    denom2 = (alpha*b-z)**2 + r**2
    e2 = (alpha*b+z)/np.sqrt(denom1) + (alpha*b-z)/np.sqrt(denom2)
    
    psi = psi_w * (E0*e0 + E1*e1 + E2*e2)
    return psi



###-----GRADIENT OF EXTERNAL MAGNETIC FLUX WRT R, dψ/dr(r,z); units = [T*m]
def external_dpsi__dr(r, z, Bw, a, b, E0, E1, E2, alpha):
    denom1 = (alpha*b-z)**2 + r**2
    denom2 = (alpha*b+z)**2 + r**2
    T0 = E0 * r
    T1 = E1 * (r/a**2) * (2*r**2 - 4*z**2)
    T2 = E2 * (a**2/2) * (r*(alpha*b-z)/(denom1**1.5) + r*(alpha*b+z)/(denom2**1.5))

    dpsi__dr = Bw * (T0 + T1 - T2)
    return dpsi__dr



###-----GRADIENT OF EXTERNAL MAGNETIC FLUX WRT Z, dψ/dr(r,z); units = [T*m]
def external_dpsi__dz(r, z, Bw, a, b, E0, E1, E2, alpha):
    denom1 = (alpha*b-z)**2 + r**2
    denom2 = (alpha*b+z)**2 + r**2
    T1 = E1 * 8*z*r**2/a**4
    T2 = E2 * ((z-alpha*b)**2/(denom1**1.5) - 1/denom1**0.5 + 1/denom2**0.5 - (alpha*b+z)**2/denom2**1.5)
    
    dpsi__dz = Bw * a**2/2 * (T2 - T1)
    return dpsi__dz



###-----EXTERNAL E AND ALPHA COEFFICIENTS; units = [\]; There's a suggest initial condition in Steinhauer
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



###-----EXTERNAL MAGNETIC FIELD AT (r=a+, z=0), B_e; just on the outside of the separatrix; units = [T]
def external_Be(Bw, a, b, Xs):
    Be__Bw = 1 + 1.46 * (3.2 + ((b/a) / (1.5-Xs))**4)**(-1)
    Be = Bw * Be__Bw
    return Be



###-----EXTERNAL FIELD OUTLINE IN SPORER; units = [T]
def sporer_Be(Xs, B0):
    Be = B0 / (1 - Xs**2)
    return Be




###----------------------------------------------------------------------------------INTERNAL FUNCTIONS
###-----STEINHAUER'S INTERNAL MAGNETIC FLUX, ψ_int(r,z); units = [T*m^2]
def internal_psi_stein(r, z, a, b, B0, B1, D0, D1):
    b0 = 1 - (r/a)**2 - (z/b)**2
    d0 = (r/a)**2 - 4*(z/a)**2
    d1 = (r/a)**4 - 12*(r*z/a/a)**2 + 8*(z/a)**4
    b1 = 1 + D0*d0 + D1*d1

    psi_int = (r**2/2) * (B0*b0 + B1*b1)
    return psi_int



###-----STEINHAUER'S INTERNAL GRADIENT OF THE MAGNETIC FLUX WRT R, dψ/dr(r,z); units = [T*m]
def internal_dpsi__dr_stein(r, z, a, b, B0, B1, D0, D1):
    b0 = 1 - 2*(r/a)**2 - (z/b)**2
    d0 = 2 * (r/a)**2 - 4 * (z/a)**2
    d1 = 3 * (r/a)**4 - 24 * (r*z/a/a)**2 + 8 * (z/a)**4
    b1 = 1 + D0*d0 + D1*d1

    dpsi__dr = (B0 * r * b0) + (B1 * r * b1)
    return dpsi__dr



###-----STEINHAUER'S INTERNAL GRADIENT OF THE MAGNETIC FLUX WRT Z, dψ/dz(r,z); units = [T*m]
def internal_dpsi__dz_stein(r, z, a, b, B0, B1, D0, D1):
    b0 = - 2 * z / b**2
    d0 = - 8 * z / a**2
    d1 = -24 * r**2 * z / a**4 + 32 * z**3 / a**4
    b1 = D0 * d0 + D1 * d1
    
    dpsi__dz = (r**2 / 2) * (B0*b0 + B1*b1)
    return dpsi__dz



###-----STEINHAUER'S INTERNAL MAGNETIC FLUX FOR E>>1, ψ_int(r,z); units = [T*m^2]; Sporer uses it as an approximation in his paper
def internal_psi_sporer(r, z, a, b, Be, Xs, f):
    eps = a / b         #Inverse elongation
    T1 = np.sqrt(3/2)
    T2 = (Xs * Be * r**2) / 2
    T3 = 1 - (r/a)**2 - (z/b)**4 + f * eps**2
    
    psi_int = T1 * T2 * T3 
    return psi_int



###-----
def internal_dBr__dz(r, z, a, b, B0, D0, D1):
    d0 = (1 / a**2)
    d1 = (3 * r**2 / z**4) - (12 * z**2 / a**4)
    b0 = 1 / b**2
    b1 = 4 * (D0 * d0 + D1 * d1)
    
    dBr__dz = r * (B0 * b0 + B1 * b1)
    return dBr__dz


###-----
def internal_dBz__dr(r, z, a, b, B0, B1, D0, D1):
    d0 = 4 * r / a**2
    d1 = 12 * (r**3 / a**4 - 4 * r * z**2 / a**4)
    b0 = -4 * r / a**2
    b1 = D0 * d0 + D1 * d1
    
    dBz__dr = B0 * b0 + B1 * b1
    return dBz__dr




###-----SPORER'S INTERNAL GRADIENT OF THE MAGNETIC FLUX WRT R, dψ/dr(r,z); units = [T*m]
def internal_dpsi__dr_sporer(r, z, a, b, Be, Xs, f):
    eps = a / b
    F = np.sqrt(3/2) * Xs * Be * r
    P = 1 - 2*(r/a)**2 - (z/b)**4 + f * eps**2
    
    dpsi__dr = F * P
    return dpsi__dr
    


###-----SPORER'S INTERNAL GRADIENT OF THE MAGNETIC FLUX WRT Z, dψ/dz(r,z); units = [T*m]
def internal_dpsi__dz_sporer(r, z, a, b, Be, Xs):
    dpsi__dz = - np.sqrt(6) * Xs * Be * r**2 * z**3 / b**4
    return dpsi__dz



###-----
def external_dBr__dz(r, z, a, b, Bw, E1, E2, alpha):
    e1 = 4 * r * Bw / a**2
    e2 = 3 * a**2 * Bw / (2 * r)
    denom_plus = (alpha*b + z)**2 + r**2
    denom_minus = (alpha*b - z)**2 + r**2
    T1 = (alpha * b - z)**3 / (denom_minus)**2.5
    T2 = (z - alpha * b) / (denom_minus)**1.5
    T3 = (alpha * b + z) / (denom_plus)**1.5
    T4 = (alpha * b + z)**3 / (denom_plus)**2.5
    Eterm =  T1 + T2 - T3 + T4
   

    dBr__dz = e1 * E1 - e2 * E2 * Eterm
    return dBr__dz



def external_dBz__dr(r, z, a, b, Bw, E2, alpha):
    p = (alpha*b + z)**2 + r**2
    m = (alpha*b - z)**2 + r**2
    numerator = (z + alpha * b) * (p**2.5 - m**2.5) 
    denominator = m**2.5 * p**2.5 
    e2 = a**4 * numerator/denominator

    dBz__dr = (Bw * r / a**2) * (E2 * e2 + 4 * E1)
    return dBz__dr



###-----FUNCTION FOR NOMINAL MAGNETIC FIELD, B_0; units = [T]
def internal_B0(a, b, Xs, Bw):
    N = 2.56 - 3.28*Xs
    D = 9.65 - 15.2*Xs + (b/a)**4
    p = 1 + N/(Xs**2 * D)    
    sr = np.sqrt(3/2)*Xs

    B0 = Bw * sr * p
    return B0



###-----SHAPE INDEX FUNCTION (crap); Eq(7a) in Steinhauer
def shape_index_N(a, b, Rs):
    N = b**2 / a / Rs
    return N



###-----SHAPE INDEX FUNCTION (crap); Eq(7b) in Steinhauer
def shape_index_N2(eps, E0, E1, E2, alpha):
    lamb = (1 + (eps/alpha)**2)**(-0.5)
    N1 = 4 * eps * E1
    N2 = 3 * alpha * lamb**5 * E2
    D1 = -eps * E1
    D2 = alpha * lamb * E2 * (2 + lamb**2)

    N = (N1 + N2) / (eps**2 * (D1 + D2))
    return N



###-----SHAPE INDEX FUNCTION (good); Eq(8) in Steinhauer
def shape_index_N3(a, b, Xs):
    N = (1 + (0.8+Xs**2) * (b/a-1)**2)**(-1)
    return N



###----FUNCTION FOR D_0 CONSTANT; units = [\]
def internal_D0(eps):
    D0 = (eps**4-8) / (4*(2+eps**2))
    return D0



###----FUNCTION FOR D_1 CONSTANT; units = [\]
def internal_D1(eps):
    D1 = - (eps**2*(4+eps**2)) / (4*(2+eps**2))
    return D1



###----FUNCTION FOR D_0 CONSTANT; units = [\]
def internal_B1(B0, D0, D1, N, eps):
    T2 = eps**2 * (N-1)
    T0 = D0 * (4+eps**2*N)
    T1 = D1 * (12 + 2*eps**2 * N)

    B1 = B0 * T2 / (T0 + T1)
    return B1




###-------------------------------------------------------------------------------------OTHER FUNCTIONS
###-----STEINHAUER'S INTERNAL PRESSURE FOR E>>1, ψ_int(r,z); units = [T*m^2]; Sporer uses it as an approximation in his paper
def pressure_sporer(Be, Xs, psi_int, a, b, f):
    eps = a / b
    Pe = Be**2 / (2*mu0)
    x = 3/2 * Xs**2
    y = np.sqrt(3/2) * Xs * 8 * psi_int / (Be*a**2)
    z = f * eps**2

    P = Pe * (1 - x + y + z)
    return P



###-----
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
        #integral_rev = cumtrapz(col_dPdr_rev, r_rev, initial=0.0)
        # By construction, integral_rev[0] = 0 at r_rev[0]=r_max, so P(r_max)=0.
        # 5b) Manual trapezoidal integration
        integral_rev = np.zeros_like(r_rev)
        for k in range(1, len(r_rev)):
            dx = r_rev[k] - r_rev[k - 1]
            avg = 0.5 * (col_dPdr_rev[k] + col_dPdr_rev[k - 1])
            integral_rev[k] = integral_rev[k - 1] + dx * avg

        # 5c) Flip back to original r‐ordering and negate to get P(r_i):
        #     integral_rev[::-1][i] = ∫_{r_i}^{r_max} dP/dr' dr'.
        #     We want P(r_i) = - ∫_{r_i}^{r_max} (dP/dr') dr'  (so that ∂P/∂r = dP/dr).
        P[:, j] = - integral_rev[::-1]

    return P



def pressure_stein(r, z, Jphi, Br, Bz):
    Nr, Nz = Jphi.shape
    
    dP__dr = Jphi * Bz          #partial of pressure wrt r [Pa/m]
    dP__dz = - Jphi * Bz        #partial of pressure wrt z [Pa/m]

    P = np.zeros((Nr, Nz))      #initialize pressure array [Pa]

    r_rev = r[::-1]             #reverse r‐axis arrays so that "index 0" in the reversed array is r_max
    dr_rev = np.diff(r_rev)     #differences along reversed array

    for j in range(Nz):
        col_dPdr = dP__dr[:, j]          # Extract column j of dP/dr and reverse it:
        col_dPdr_rev = col_dPdr[::-1]

        # Cumulatively integrate col_dPdr_rev from r_max downward:
        #    cumtrapz(y, x, initial=0) returns an array Y of length len(x),
        #    where Y[k] = ∫_{x[0]}^{x[k]} y(x') dx'.
        #
        # Here x = r_rev, y = col_dPdr_rev, and x[0] = r_rev[0] = r_max.
        # So cumtrapz(col_dPdr_rev, r_rev, initial=0) gives:
        #   integrated[k] = ∫_{r_max}^{ r_rev[k] } dP/dr * d(r').
        # Because r_rev is decreasing, this is effectively ∫_{r_rev[k]}^{r_max}.
        #
        integral_rev = cumtrapz(col_dPdr_rev, r_rev, initial=0.0)
        # By construction, integral_rev[0] = 0 at r_rev[0] = r_max.

        # Now flip back to the original r‐ordering:
        #   P[:, j] = - integral from r_i to r_max = -integral_rev[ reversed index ]
        P[:, j] = - integral_rev[::-1]

    return P



###-----FUNCTION FOR RETURNING THE (r,z) COORDINATES OF A CONTOUR MAGNITUDE OF YOUR CHOOSING; units = [m]; can also return multiple contours by setting return_all=True
def get_flux_contours(psi, R, Z, psi_level_mag, return_all=False):
    # … earlier code …

    # 1) Build a tiny OFF‐SCREEN figure, extract contours, then close it:
    plt.ioff()                     # turn off interactive showing
    fig, ax = plt.subplots(figsize=(0.1,0.1))  
    cs = ax.contour(R, Z, psi, levels=[psi_level_mag])
    plt.close(fig)                 # immediately close it so nothing pops up

    # 2) Now extract the contour segments from cs:
    try:
        idx = list(cs.levels).index(psi_level_mag)
    except ValueError:
        raise RuntimeError(f"No ψ={psi_level_mag} level found in cs.levels={cs.levels}")
    segs = cs.allsegs[idx]
    if not segs:
        raise RuntimeError(f"No ψ={psi_level_mag} contour found in the domain.")

    # 3) Convert each Nx2 array into (r_i, z_i) loops exactly as you had:
    loops = []
    for seg in segs:
        verts = np.asarray(seg)    # shape=(Npts,2)
        r_i = verts[:,0].copy()
        z_i = verts[:,1].copy()
        loops.append((r_i, z_i))

    if return_all:
        return loops

    longest = max(loops, key=lambda pair: pair[0].shape[0])
    return longest


# ===========================
# Example usage:

# Suppose you have psi, R, Z already defined on your mesh:
#    psi = ...   # 2D array shape (Nr,Nz)
#    R, Z = np.meshgrid(r, z, indexing='ij')

# To get only the longest loop at psi = 1.0:
#r0, z0 = get_flux_contours(psi, R, Z, psi_level_mag=1.0, return_all=False)

# To get all disjoint loops at psi = 1.0:
#all_loops = get_flux_contours(psi, R, Z, psi_level_mag=1.0, return_all=True)
# all_loops is now something like [(r1,z1), (r2,z2), …]

# If you just want to plot them:
#for (r_i, z_i) in all_loops:
#    plt.plot(r_i, z_i, '-k', linewidth=1)
#
#plt.gca().set_aspect('equal')
#plt.show()






#######################################################################################################
#######################################################################################################
#################################                  MAIN                    ############################
#######################################################################################################
#######################################################################################################
Lconv = 1e-2            #length conversion used so that displayed lengths in the code are in [cm]
###------------------------------------------------------------------------------------------PARAMETERS
Xs = 0.73            #normalized separatrix ratio [\]
E = 4.5

zLen = 2.5                  #[cm]
liner_top = zLen*Lconv       #[m]
liner_bot = -0.3*Lconv      #[m]
liner_bot = -zLen*Lconv      #[m]
h = liner_top - liner_bot   #[m]

Rw = 0.6*Lconv         #flux-conserving wall radius [m]
Rc = Rw                 #coil radius; assumed to by Rw here [m]
Rs = Xs * Rw            #separatrix radius [m]
Zs = E * Rs
#R0 = Rs / np.sqrt(2)    #O-point [m]


a = Rs                  #FRC semi-minor radius [m]
b = Zs                  #FRC semi-major radius [m]

sig = 1.5               #flare parameter; adjustable parameter that's fixed for Steinhauer's paper
f = 1.5                 #internal psi error factor for Sporer's approximation
eps = a / b             #inverse elongation

B0 = 4.04*1.41
#B0 = 30
Be = B0 / (1-Xs**2)
Bw = Be                 #Magnetic field at the midplane at the wall [T] from Steinhaur
B00 = Bw                #Sporer vacuum field [T] still working out how this relates to Steinhauer
Zi = 1                  #ion charge number [\]
m_ave = (3.016*amu + 2.014*amu) / 2
T_eV = 460              #Temperature [eV]
T = T_eV * eV/kb        #Temp number in [K]
print("m_ave = ", m_ave)

if Zs > 0.75*h/2:
    print("FRC is too long: Zs > h/2")
    print("               {:.2e} [m] > {:.2e} [m]".format(Zs, h/2))
    print("Reduce E or Xs")

###----------------------------------------------------------------------------------------DOMAIN SETUP
domx = 1.0                                  #weight to extend the domain; helps show psi=1 curves better
Rd = Rw * domx                              #half-length of computational domain in r-dir [m]
Zd = h/2 * domx                             #half-length of computational domain in z-dir [m]

dr = 0.001*Lconv                            #mesh fidelity in r-dir [m]
dz = 0.001*Lconv                            #mesh fidelity in z-dir [m]

r = np.arange(0, Rd + dr, dr)
z = np.arange(-Zd, Zd + dz, dz)
R, Z = np.meshgrid(r, z, indexing='ij')

R_cc = 0.5 * (R[:-1, :-1] + R[1:, 1:])
Z_cc = 0.5 * (Z[:-1, :-1] + Z[1:, 1:])
print("Domain Size: {:.3e}m x {:.3e}m".format(2*Rd, 2*Zd))
print("Domain Size: Rd = {:.3e} m     Zd = {:.3e}".format(Rd, Zd))

###-------------------------------------------------------------------------------------EXTERNAL REGION
Eguess = [Bw, Bw/3, Bw/5, 0.9]              #initial guess for E parameters
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

    psi_ext = external_psi(R, Z, B00, a, b, E0, E1, E2, alpha)                  #external magnetic flux
    dpsi__dr_ext = external_dpsi__dr(R, Z, Bw, a, b, E0, E1, E2, alpha)         #[T*m^2]; gradient of
    dpsi__dz_ext = external_dpsi__dz(R, Z, Bw, a, b, E0, E1, E2, alpha)         #external magnetic flux
    Br_ext = -(1/R) * dpsi__dz_ext          #radial magnetic field [T]          #[T*m]
    Bz_ext = (1/R) * dpsi__dr_ext           #axial magnetic field [T]
    #check ^these^ signs

N = shape_index_N3(a, b, Xs)                #shape index [\]; N=0 (racetrack), N=1 (ellipse/Hills vortex)
D0 = internal_D0(eps)                       #D_0 constant [\]
D1 = internal_D1(eps)                       #D_1 constant [\]
B0 = internal_B0(a, b, Xs, Bw)              #nominal magnetic field [T]
B1 = internal_B1(B0, D0, D1, N, eps)        #"other field component"; not pictured [T]
Be = external_Be(Bw, a, b, Xs)              #External magnetic field just outside the separatrix [T]
Bee = sporer_Be(Xs, B00)                    #Sporer's external magnetic field [T]



###-------------------------------------------------------------------------------------INTERNAL REGION
model = "sporer"     #choose between Steinhauer's full solution (stein), or the approx for E>>1 (sporer)

if model=="stein":
    psi_int = internal_psi_stein(R, Z, a, b, B0, B1, D0, D1)                #internal magnetic flux
    Br_int = (1/R) * internal_dpsi__dz_stein(R, Z, a, b, B0, B1, D0, D1)    #[T*m^2]; internal radial and
    Bz_int = -(1/R) * internal_dpsi__dr_stein(R, Z, a, b, B0, B1, D0, D1)   #axial magnetic fields [T]
elif model=="sporer":
    psi_int = internal_psi_sporer(R, Z, a, b, Bw, Xs, f)                   
    Br_int = (1/R) * internal_dpsi__dz_sporer(R, Z, a, b, Bw, Xs)
    Bz_int = -(1/R) * internal_dpsi__dr_sporer(R, Z, a, b, Bw, Xs, f)


B_int = np.sqrt(Br_int**2 + Bz_int**2)
print(np.max(B_int))
print(B_int)
ave_Bi = np.mean(B_int)
print("<B_i> = {:.3f} [T]".format(ave_Bi))
#Finds indexing for the midplane Bz check
mid_z = np.argmin(np.abs(z))
mid_r = int(len(r)/2)

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


rSep, zSep = get_flux_contours(psi, R, Z, 0)            #gets a contour of the separatrix (psi=0)
r0_comb, z0_comb = get_flux_contours(psi, R, Z, 0)


###-----Debugging for negative region between psi_ext and psi_int
# find every negative ψ
neg_vals = psi[psi < 0]

# print them
#print("All negative psi values:")
#print(neg_vals)
#Finds indexing for the midplane Bz check
mid_z = np.argmin(np.abs(z))
#mid_r = int(len(r)/2)
#zidx = np.where(np.isclose(z,Z0))[0][0]
#print(zidx, psi[0,zidx])




# index of z ≈ 0 (column in Bz)
mid_z = int(np.argmin(np.abs(z)))

# keep r from 0 → rmax
mask = r >= 0

# 1-D r and matching Bz along z=0
rSlice = r[mask]            # shape (Nr_pos,)
BSlice = Bz[mask, mid_z]    # shape (Nr_pos,)
idxB = np.nanargmin(np.abs(BSlice))
r_zero = rSlice[idxB]
bz_at_min = BSlice[idxB]
print(f"min|Bz| at r={r_zero:.6g}, Bz={bz_at_min:.3e}")
mid_z = int(np.argmin(np.abs(z)))
r_col  = R[:, mid_z]         # 2-D column of r values at z=0
bz_col = Bz[:, mid_z]
mask   = r_col >= 0
rSlice = r_col[mask]
BSlice = bz_col[mask]






###----------------------------------------------------------------------------------------------OUTPUT
print("FRC PARAMETERS---------------------------------------------------------------------------------")
print("\t Rw = {:.4f} [cm] (Conducting Wall Radius)".format(Rw/Lconv))
print("\t Rs = {:.4f} [cm] (Spearatrix Radius)".format(Rs/Lconv))
print("\t Xs = {:.4f} (Normalized Separatrix Radius)".format(Xs))
#print("\t R0 = {:.4f} [cm] (O-point)".format(R0/Lconv))
print(" ")

print("\t E = {:.4f} (FRC Elongation)".format(1/eps))
print("\t eps = {:.4f} (Inverse Elongation)".format(eps))
print("\t a = {:.4f} = [cm] (FRC Radius)".format(a/Lconv))
print("\t b = {:.4f} [cm] (FRC Half Length)".format(b/Lconv))
print("\t sig = {:.2f} (Adjustable Parameter)".format(sig))
print(" ")


print("\t N = {:.4f} (Shape Index; 0=racetrack  1=ellipse)".format(N))
print("\t E0 = {:.4f} ".format(E0))
print("\t E1 = {:.4f} ".format(E1))
print("\t E2 = {:.4f} ".format(E2))
print("\t alpha = {:.4f} (0 < alpha < 1)".format(alpha))
print(" ")

print("\t B0 = {:.3f} [T]".format(B0))
print("\t B1 = {:.3f} [T]".format(B1))
print("\t Bw = {:.3f} [T]".format(Bw))
print("\t Be = {:.3f} [T]".format(Be))
print("\t <B_i> = {:.3f} [T]".format(ave_Bi))
print("\t D0 = {:.4f} ".format(D0))
print("\t D1 = {:.4f} ".format(D1))
print(" ")



# — build a single string with all your FRC parameters —
summary_text = (
    "FRC PARAMETERS\n"
    f"Rc = {Rw/Lconv:.4f} [cm] (Conducting Wall Radius)\n"
    f"Rs = {Rs/Lconv:.4f} [cm] (Separatrix Radius)\n"
    f"Xs = {Xs:.4f} (Normalized Separatrix Radius)\n"
#    f"R0 = {R0/Lconv:.4f} [cm] (O-point Radius)\n\n"
    f"E  = {1/eps:.4f} (FRC Elongation)\n"
    f"eps = {eps:.4f} (Inverse Elongation)\n"
    f"a  = {a/Lconv:.4f} [cm] (FRC Radius)\n"
    f"b  = {b/Lconv:.4f} [cm] (FRC Half Length)\n"
    f"sig = {sig:.2f} (Adjustable Parameter)\n\n"
    f"N  = {N:.4f} (Shape Index; 0=racetrack 1=ellipse)\n"
    f"E0 = {E0:.4f}\n"
    f"E1 = {E1:.4f}\n"
    f"E2 = {E2:.4f}\n"
    f"alpha = {alpha:.4f} (0 < alpha < 1)\n\n"
    f"B0 = {B0:.3f} [T]\n"
    f"B1 = {B1:.3f} [T]\n"
    f"Bw = {Bw:.3f} [T]\n"
    f"Be = {Be:.3f} [T]\n"
    f"D0 = {D0:.4f}\n"
    f"D1 = {D1:.4f}"
)



###------------------------------------------------------------------------------------------------DIV(B)
dBr__dr, dBr__dz = np.gradient(Br, r, z, edge_order=2)  #gradient of Br [T/m]
dBz__dr, dBz__dz = np.gradient(Bz, r, z, edge_order=2)  #gradient of Bz [T/m]

#add in my functions here and compare to see new modB graphs

divB = dBr__dr + Br/R + dBz__dz                         #divergence of B [T/m]
divB_scaled = np.abs(divB) / (Bw/Rs)                    #scaled divergence of B [\]



###-----Debugging for the numerical anomalies in the divB calcs
idx = 0
idy = idx
print("Divergence of B:")
print("\tdBr__dr \t Br/R \t\t dBz__dz \t divB \t\t divB_scaled")
print(
    f"\t{dBr__dr[idx,idy]} "
    f"{Br[idx,idy]/R[idx,idy]} "
    f"{dBz__dz[idx,idy]} "
    f"{divB[idx,idy]} "
    f"{divB_scaled[idx,idy]}"
)
print("\t divB_scaled max and min", divB_scaled.max(), divB_scaled.min())





###-----------------------------------------------------------------------------------CURRENT DENSITY
J = (1 / mu0) * (dBr__dz - dBz__dr)     #current density [A/m^2]
Jphi = J

print("===== CURRENT DENSITY =====")
print(np.max(Jphi))



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



###-----Debugging - Error Check
# Compute numerical partials of your reconstructed P(r,z):
dPdr_check, dPdz_check = np.gradient(P, r, z, edge_order=2)

# Compare to the original dP__dr, dP__dz:
print("Pressure:")
print("\tMax Pressure = {:.3e} [Pa]".format(P_int.argmax()))
print("\t Max |dPdr_check - dP__dr| =", np.nanmax(np.abs(dPdr_check - dP__dr)))
print("\t Max |dPdz_check - dP__dz| =", np.nanmax(np.abs(dPdz_check - dP__dz)))





###------------------------------------------------------------------------------------NUMBER DENSITY

print(kb, T)
n_edges = P / (kb * T)

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
n_cci = n_cc / 2
n_cce = n_cc / 2

#####---NUMBER DENSITY DEBUG
print("NUMBER DENSITY DEBUG")
n_max_edges = n_edges.max()
n_max_cc = n_cc.max()


print("\tMax Number Density (n_edges.argmax) = {:.3e} [m^-3]".format(n_max_edges))
print("\tMax Number Density (n_cc.argmax) = {:.3e} [m^-3]".format(n_max_cc))
print("\tMax Mass Density (n_edges.argmax) = {:.3e} [kg/m^3]".format(m_ave * n_max_edges))
print("\tMax Mass Density (n_cc.argmax) = {:.3e} [kg/m^3]".format(m_ave * n_max_cc))
print("P_int = ", P_int.argmax(), "P_max = ", P.argmax())



r_cc = 0.5*(r[:-1] + r[1:])
z_cc = 0.5*(z[:-1] + z[1:])

r0_idx = np.searchsorted(r_cc, 0.0)
z0_idx = np.abs(z_cc - 0.0).argmin()

r_cm   = r_cc[r0_idx:] * 1e2                     # r ≥ 0 in cm
n_mid  = n_cc[r0_idx:, z0_idx]                   # midplane number density
ni_mid = n_mid / 2
rho_mid = m_ave * ni_mid                          # mass density [kg m^-3] if m_ave in kg



X = R_cc*1e2          # r in cm (cell centers)
Y = Z_cc*1e2          # z in cm

rho_cc = n_cci * m_ave       #mass density [kg/m^3]
rho_cc = rho_cc * (1e-3)    #mass density [g/cm^3]

vmin = 1.0e-7
vmax = 1.0e-3
print("n_max = {:.3e} [m^-3]".format(n_mid.max()))
'''
###-----------------------------------MIDPLANE SLICES
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.5, 6.5), constrained_layout=True)

# --- top: number density ---
ax1.plot(r_cm, n_mid)
ax1.set_ylabel(r'$n_{\mathrm{cc}}(r,0)\;[{\rm m^{-3}}]$')
ax1.set_title('Cell-Centered Midplane Density (r ≥ 0)')
ax1.grid(True)

# --- bottom: mass density ---
ax2.plot(r_cm, rho_mid*(1e-3))
ax2.set_xlabel('r [cm]')
ax2.set_ylabel(r'$\rho(r,0)=m_{\rm ave}\,n_{\rm cc}\;[{\rm g\,cm^{-3}}]$')
ax2.set_title('Midplane Mass Density (r ≥ 0)')
ax2.grid(True)
#plt.show()
#plt.close()
'''
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# --- data (X, Y, rho_cc) are assumed already defined ---

#r = np.arange(0, Rd + dr, dr)
#z = np.arange(-Zd, Zd + dz, dz)
#R, Z = np.meshgrid(r, z, indexing='ij')
#R_cc = 0.5 * (R[:-1, :-1] + R[1:, 1:])

# limits
USE_LOG = True  # True -> LogNorm, False -> linear

# copy cmap and set the "under"/"over" colors to be exactly the end colors
cmap = plt.cm.get_cmap('turbo').copy()
cmapBz = plt.cm.get_cmap('Spectral_r').copy()
cmap.set_under(cmap(0))          # below vmin -> same as lowest color
cmap.set_over(cmap(cmap.N - 1))  # above vmax -> same as highest color

# norm + levels
if USE_LOG:
    norm   = colors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.geomspace(vmin, vmax, 100)
    Zplot  = np.where(rho_cc > 0, rho_cc, np.nan)  # LogNorm needs > 0
    Zplot = np.where(rho_cc <= 0.0, vmin/10.0, rho_cc)   # <-- key line
    #Zplot  = np.where(rho_cc = 0, vmin/10, np.nan)  # LogNorm needs > 0
else:
    norm   = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    levels = np.linspace(vmin, vmax, 100)
    Zplot  = rho_cc

plt.figure(figsize=(6, 8))

# extend='both' ensures regions <vmin and >vmax are filled;
# with set_under/set_over equal to the endpoints, they appear clamped.
contour = plt.contourf(
    X, Y, Zplot, levels=levels, cmap=cmap, norm=norm, extend='both'
)
levelsBz = np.linspace(-20,20,100)
Bzcont = plt.contourf(
    -R*100, Z*100, Bz, levels=levelsBz, cmap=cmapBz)

cbar = plt.colorbar(contour, extend='both', label='Mass Density, $[g/cm^{3}]$')
if USE_LOG:
    ticks = np.geomspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1e}" for t in ticks])
caxBz = plt.gcf().add_axes([0.05, 0.15, 0.03, 0.70])
cbarBz = plt.colorbar(Bzcont, extend='both', cax=caxBz, label='$B_z$, $[T]$')
ticksBz = np.geomspace(-20, 20, 100)
cbarBz.set_ticks(ticksBz)
cbarBz.set_ticklabels([f"{t:.1f}" for t in ticksBz])
cbarBz.ax.set_position([0.05, 0.15, 0.03, 0.70])
cbarBz.ax.yaxis.set_ticks_position('left')


plt.xlabel('r [cm]')
plt.ylabel('z [cm]')
plt.xlim(-Rd/Lconv, Rd/Lconv)
plt.ylim(-Zd/Lconv, Zd/Lconv)
plt.title('Cell-Centered Mass Density')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
#plt.close()
sys.exit()
'''
#===============================================================================================
# ----------------- FIGURE + MAIN AXES -----------------
fig = plt.figure(figsize=(6, 8))

from matplotlib import colors
# Main plot axes (leave room on both sides for colorbars)
ax = fig.add_axes([0.18, 0.12, 0.64, 0.80])   # [left, bottom, width, height]

# Colorbar axes: Bz on LEFT, density on RIGHT
caxBz  = fig.add_axes([0.2, 0.12, 0.04, 0.80])
caxRho = fig.add_axes([0.7, 0.12, 0.04, 0.80])

cmap = plt.cm.get_cmap('turbo').copy()
cmapBz = plt.cm.get_cmap('Spectral_r').copy()
cmap.set_under(cmap(0))          # below vmin -> same as lowest color
cmap.set_over(cmap(cmap.N - 1))  # above vmax -> same as highest color
USE_LOG = True  # True -> LogNorm, False -> linear
if USE_LOG:
    norm   = colors.LogNorm(vmin=vmin, vmax=vmax)
    levels = np.geomspace(vmin, vmax, 100)
    Zplot  = np.where(rho_cc > 0, rho_cc, np.nan)  # LogNorm needs > 0
    Zplot = np.where(rho_cc <= 0.0, vmin/10.0, rho_cc)   # <-- key line
    #Zplot  = np.where(rho_cc = 0, vmin/10, np.nan)  # LogNorm needs > 0
else:
    norm   = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    levels = np.linspace(vmin, vmax, 100)
    Zplot  = rho_cc

# ----------------- PLOTTING -----------------
# Density (main) filled contours
rho_cont = ax.contourf(
    X, Y, Zplot,
    levels=levels,
    cmap=cmap,
    norm=norm,
    extend='both'
)

# Bz (overlay) filled contours
levelsBz = np.linspace(-20, 20, 100)
bz_cont = ax.contourf(
    -R*100, Z*100, Bz,
    levels=levelsBz,
    cmap=cmapBz,
    extend='both',
    alpha=0.85  # tweak: 0.6-0.9 so density still visible
)

# ----------------- COLORBARS -----------------

# LEFT: Bz
cbarBz = fig.colorbar(bz_cont, cax=caxBz, extend='both')
cbarBz.set_label(r'$B_z$ [T]')
cbarBz.ax.yaxis.set_ticks_position('left')
cbarBz.ax.yaxis.set_label_position('left')

# your tick choice (keep / adjust as you like)
ticksBz = np.linspace(-20, 20, 5)
cbarBz.set_ticks(ticksBz)
cbarBz.set_ticklabels([f"{t:.1f}" for t in ticksBz])

# RIGHT: density
cbar = fig.colorbar(rho_cont, cax=caxRho, extend='both')
cbar.set_label(r'Mass Density, $[g/cm^{3}]$')

if USE_LOG:
    ticks = np.geomspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1e}" for t in ticks])

# ----------------- AXES LABELS/LIMITS -----------------
#caxRho.remove()
ax.set_xlabel('r [cm]')
ax.set_ylabel('z [cm]')
ax.set_xlim(-Rd/Lconv, Rd/Lconv)
ax.set_ylim(-Zd/Lconv, Zd/Lconv)
ax.set_aspect('equal', adjustable='box')

# (optional) title
#ax.set_title('Cell-Centered Mass Density (with $B_z$ overlay)')

plt.savefig('fig1.png', dpi=dpi_res)
plt.show()
# plt.close(fig)   # uncomment if running in a loop





























#===============================================================================================

###------------------------------------------KINETICS CALCS
def sbar_from_arrays(r, rho_i, R0, Rs, rho_min=None):
    """
    r, rho_i : 1D arrays sampled on the same grid
    R0, Rs   : integration limits (Rs > R0)
    rho_min  : optional floor to avoid division by ~0 (e.g., 1e-9)
    """
    r = np.asarray(r, dtype=float)
    rho_i = np.asarray(rho_i, dtype=float)
    if r.ndim != 1 or rho_i.ndim != 1 or r.size != rho_i.size:
        raise ValueError("r and rho_i must be 1D arrays of equal length.")
    if not (Rs > R0):
        raise ValueError("Require Rs > R0.")

    # make sure r is strictly increasing
    if r[0] > r[-1]:
        r = r[::-1]
        rho_i = rho_i[::-1]

    # clip or floor rho if desired
    if rho_min is not None:
        rho_i = np.maximum(rho_i, rho_min)

    # Interpolate rho at the exact boundaries R0, Rs
    # (assumes R0 and Rs lie within the span of r)
    if not (r[0] <= R0 <= r[-1] and r[0] <= Rs <= r[-1]):
        raise ValueError("R0 and Rs must lie within the r-range.")

    rho_R0 = np.interp(R0, r, rho_i)
    rho_Rs = np.interp(Rs, r, rho_i)

    # Build subgrid including the exact endpoints
    mask = (r >= R0) & (r <= Rs)
    r_seg = r[mask]
    rho_seg = rho_i[mask]

    # Ensure endpoints are present (insert if R0/Rs fall between samples)
    if r_seg.size == 0 or r_seg[0] > R0:
        r_seg = np.insert(r_seg, 0, R0)
        rho_seg = np.insert(rho_seg, 0, rho_R0)
    elif r_seg[0] < R0:  # rare, but keep consistent
        r_seg[0] = R0; rho_seg[0] = rho_R0

    if r_seg[-1] < Rs:
        r_seg = np.append(r_seg, Rs)
        rho_seg = np.append(rho_seg, rho_Rs)
    elif r_seg[-1] > Rs:
        r_seg[-1] = Rs; rho_seg[-1] = rho_Rs

    # Integral
    integrand = r_seg / rho_seg
    I = np.trapezoid(integrand, r_seg)
    return I / Rs

v_thi = np.sqrt(3 * kb * T / m_ave)             #ion thermal velocity [m/s]
v_perp = v_thi                                  #ion perpendicular velocity [m/s]
rho_ie = gyroRadius(m_ave, v_perp, Zi, Be)      #ion gyroradius [m]
ni = ni_mid.max()                                #using the max density (O-point) as the reference density [m^-3]
print("plasma frequency-----",ni, ee, m_ave, eps0)
om_pi = np.sqrt((ni * ee**2) / (m_ave * eps0))  #plasma frequency [rad/s]
delt_i = ccc / om_pi                            #ion inertial length (skin depth) [m]


idxB = np.argmin(np.abs(BSlice))
print("idxB =", idxB)
r_zero = rSlice[idxB]              # same units as r
r_zero_norm = rSlice[idxB]/Lconv   # if you want the normalized value you plot
R0 = r_zero

rho_iint = gyroRadius(m_ave, v_perp, Zi, B_int)
rho_iintave = np.mean(rho_iint)
rho_iSlice = gyroRadius(m_ave, v_perp, Zi, BSlice)
#print(rSlice)
#print(BSlice)

###-----FOUR s PARAMETERS
Sstar = Rs / delt_i
S = R0 / rho_ie
s = Rs / (4 * rho_iintave)
sbar = sbar_from_arrays(rSlice, rho_iSlice, R0, Rs, rho_min=1e-8) 

#print(mid_r, mid_z)
#rSlice = r[:, mid_z]
#BSlice = Bz[mid_r:, mid_z]


print("KINETICS-------------------")
print("\t Ion Thermal Velocity, v_th,i = {:.3e} [m/s]".format(v_thi))
print("\t Ion Gyroradius (at wall magnetic field), rho_i,w = {:.3f} [mm]".format(rho_ie*1e3))
print("\t Average Internal Ion Gyroradius, <rho_i,int> = {:.3f} [mm]".format(rho_iintave*1e3))
print("\t Ion Inertial Length, delta_i = {:.3f} [mm]".format(delt_i*1e3))
print("\t FRC Kinetic S = {:.3f}".format(S)) 
print("\t FRC Kinetic S* = {:.3f}".format(Sstar)) 
print("\t FRC Kinetic s = {:.3f}".format(s)) 
print("\t FRC Kinetic sbar = {:.3f}".format(sbar)) 
print("\t Magnetic Axis (O-point), R0 = {:.3f} [mm]".format(R0*1e3))
print(f"\t min|Bz| at idxB={idxB}: r={r_zero:.6g} [m] (norm {r_zero_norm:.6g}), Bz={BSlice[idxB]:.3e} [T]")



