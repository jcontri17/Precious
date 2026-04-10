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


def _debug():
    t0code_frc05 = time.perf_counter()

if __name__ == "__main__":
    _debug()
    print("frc05_2D_SteinhauerFunctions executed!")



