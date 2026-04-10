#FRC 2D Equilibrium         Jeffrey D. Contri; jcontri@uw.edu           2025_08_30 - 
#Ref:   Cerfon and Freidberg (2010) DOI: 10.1063/1.3328818

#################################################
#####               IMPORTS                 #####
#################################################
import numpy as np
import matplotlib.pyplot as plt
import cmath
import time
import sys
from FRC2D_GSfunctions import *
from functions4plasma import *
from plottingParameters import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D



t0code02 = time.perf_counter()
#################################################
#####               FUNCTIONS               #####
#################################################

###----------------------------------------------------------------------------------Phi_h(x,y) MATRIX
def homog_varphi_Phi_h(xy):
    Phi_h = np.array([
        [varphi_1(*xy[0]),  varphi_2(*xy[0]),  varphi_4(*xy[0]),  varphi_6(*xy[0])],
        [varphi_1(*xy[1]),  varphi_2(*xy[1]),  varphi_4(*xy[1]),  varphi_6(*xy[1])],
        [varphi_1yy(*xy[2]), varphi_2yy(*xy[2]), varphi_4yy(*xy[2]), varphi_6yy(*xy[2])],
        [varphi_1xx(*xy[3]), varphi_2xx(*xy[3]), varphi_4xx(*xy[3]), varphi_6xx(*xy[3])]
    ])

    return Phi_h



###---------------------------------------------------------------------------------Phi_BC(x,y) MATRIX
def boundary_condition_Phi_BC(xy, N1, N3):
    Phi_BC = np.zeros((4, 4), dtype=float)

    Phi_BC[2, :] = -N1 * np.array([
        varphi_1x(*xy[2]), varphi_2x(*xy[2]), varphi_4x(*xy[2]), varphi_6x(*xy[2])])

    Phi_BC[3, :] = -N3 * np.array([
        varphi_1y(*xy[3]), varphi_2y(*xy[3]), varphi_4y(*xy[3]), varphi_6y(*xy[3])])
    
    return Phi_BC



###------------------------------------------------------------------------Varphi_p(x,y) COLUMN VECTOR
def vector_particularSol(xy, A):
    x0  = xy[0][0]
    x1  = xy[1][0]
    x2  = xy[2][0]
    x3  = xy[3][0]

    varphi_P = np.array([[varphi_p(x0, A)], [varphi_p(x1, A)], [varphi_pyy(x2, A)], [varphi_pxx(x3, A)]])
    return varphi_P



###---------------------------------------------------------------------Varphi_p_BC(x,y) COLUMN VECTOR
def boundary_condition_particularSol(xy, A, N1, N3):
    varphi_bc_P = np.zeros(4, dtype=float)

    x2  = xy[2][0]
    x3  = xy[3][0]

    varphi_bc_P[2]  = -N1 * varphi_px(x2, A)
    varphi_bc_P[3]  = -N3 * varphi_py(x3, A)

    return varphi_bc_P.reshape(4,1)




def zero_contour(X, Y, Z, tol=1e-12):
    """
    Returns the (x, y) coordinates where Z == 0 within a small tolerance.

    Parameters
    ----------
    X, Y, Z : 2D numpy arrays of identical shape
        Meshgrid arrays representing coordinates and corresponding values.
    tol : float, optional
        Numerical tolerance for equality, since exact zero is rare.

    Returns
    -------
    xy_points : ndarray of shape (N, 2)
        Each row contains (x, y) coordinates where Z ≈ 0.
    """
    if not (X.shape == Y.shape == Z.shape):
        raise ValueError("X, Y, and Z must all have the same shape.")
    
    mask = np.isfinite(Z) & (np.abs(Z) <= tol)
    x_points = X[mask]
    y_points = Y[mask]
    xy_points = np.column_stack((x_points, y_points))
    return xy_points


def print_tabbed(label, A, tabs=1):
    indent = "\t" * tabs
    print(f"{indent}{label} {A.shape}:")
    s = str(A)
    s = indent + s.replace("\n", "\n" + indent)
    print(s, "\n")



#################################################
#####               DOMAIN                  #####
#################################################
E = 10                       #Elongation [\]

###-----FRC-specific Defining Parameters
eps = 1                 #inverse aspect ratio (a/Rs); a=torus minor radius~Rs/4 (Tuszewski '90)
delta = 1               #triangulation
A = 0                   #Solov'ev constant for current profile
sig_sep = 1.1           #
alpha = np.asin(delta)  #triangularity characteristic angle


xsep = 1 - sig_sep * delta * eps
ysep = - sig_sep * E * eps

theps = 1e-6 #[rad] helps avoid singularity in ln(x) in varphi_P
theta_min = -np.pi / 2                                  #minimum parameterized angle [rad]
theta_max = np.pi/2                                     #maximum parameterized angle [rad]
Nth = 199                                               #number of discretizations minus 1
dth = (theta_max - theta_min) / Nth                     #differential parameterized angle [rad]
th = np.arange(theta_min, theta_max + dth, dth)         #paratmeterized angle array [rad]

x = 2 * np.cos(th)          #unitless radial coordinate [\]
y = E * np.sin(th)          #unitless axial coordinate [\]

X, Y = np.meshgrid(x, y)



########################################################
#####       SOLVING FOR c_i AND VARPHI(X,Y)        #####
########################################################

###-----Boundary Condition Coordinates (see Fig. 1)
xy_full = [
    (1+eps, 0),              # Row 1
    (1-eps, 0),              # Row 2
    (1-delta*eps, E*eps),    # Row 3
    (xsep, ysep),            # Row 4
    (1+eps, 0),              # Row 5
    (1-eps, 0),              # Row 6
    (1-delta*eps, E*eps),    # Row 7
    (xsep, ysep),            # Row 8
    (xsep, ysep),            # Row 9
    (1+eps, 0),              # Row 10
    (1-eps, 0),              # Row 11
    (1-delta*eps, E*eps)     # Row 12
]


###-----Boundary Conditions for FRC Solution
xyFRC = [
    (2, 0),
    (0, E),
    (2, 0),
    (0, E)
]


###-----FRC Curvature Constants for Boundary Conditions
N1 = -2 / E**2
N3 = -E / 4


###-----Setting up Solution Matrices and Implementing Boundary Conditions
Phi_h = homog_varphi_Phi_h(xyFRC)
Phi_bc = boundary_condition_Phi_BC(xyFRC, N1, N3)
varphi_P = vector_particularSol(xyFRC, A)
varphi_P_bc = boundary_condition_particularSol(xyFRC, A, N1, N3)


###-----Solving for the c_i Constants
c_vec = np.linalg.solve((Phi_bc - Phi_h), (varphi_P - varphi_P_bc))

c_h = np.zeros(12)
c_h[0] = c_vec[0].item()    #c1
c_h[1] = c_vec[1].item()    #c2
c_h[3] = c_vec[2].item()    #c4
c_h[5] = c_vec[3].item()    #c6


###-----Solving for the Scaled Flux, varphi(x,y)
varphi = np.zeros((len(x), len(y)))

for i in range(len(x)):
    for j in range(len(y)):
        varphi[i,j] = varphi_sol(x[i], y[j], A, c_h)



#######################################################################
###############################   PLOTTING   ##########################
#######################################################################
xpad = 1.5
ypad = 1.0
nLevs = 11
axisFontSize = 14
titleFontSize = 18
colors = plt.cm.rainbow(np.linspace(0, 1, nLevs))



#####============================================SCALED================================================
#####-------------------------------------------------------------------------------------------UPRIGHT
# Reflect across the y-axis (x -> -x) and stitch the two halves.
if x[0] == 0:
    x_full = np.concatenate((-x[::-1], x[1:]))             # avoid duplicate 0
    Z_full = np.concatenate((varphi[::-1, :], varphi[1:, :]), axis=0)
else:
    x_full = np.concatenate((-x[::-1], x))
    Z_full = np.concatenate((varphi[::-1, :], varphi), axis=0)
X, Y = np.meshgrid(x_full, y, indexing='ij')  # X,Y shapes (2*Nx-(x[0]==0), Ny)

# (Optional) mask non-finite values so plotting skips them
Z = np.where(np.isfinite(Z_full), Z_full, np.nan)
Z = np.where(Z < 0, Z, np.nan)  # (keep your original filtering logic if needed)
vmin = np.nanmin(Z_full)   # or whatever array you contour

fig, ax = plt.subplots(figsize=(7, 8))
cf = ax.contourf(X, Y, Z, levels=nLevs, cmap='viridis')  # change levels/cmap as you like
cs = ax.contour(X, Y, Z, levels=nLevs, colors=colors, linewidths=1.0)

levels = np.atleast_1d(cs.levels)
handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(nLevs)]
labels  = [fr'$\varphi={lev:.2f}$' for lev in levels]

ax.legend(handles, labels, title=r'Levels',
          loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=False, borderaxespad=0.0)
fig.text(0.8, 0.7, rf'$\varphi_{{\min}}={vmin:.3g}$',
         ha='center', va='top', fontsize=titleFontSize-4)
plt.tight_layout(rect=[0, 0, 0.82, 1])

# --- Add Z = 0 contour line in bold red (no .collections access) ----
Z = np.where(np.isfinite(Z), Z, np.nan)
c0 = ax.contour(X, Y, Z, levels=[0.0], colors='crimson',linewidths=1.0, linestyles='-', zorder=10)

#line_proxy = Line2D([0], [0], color='crimson', lw=3, label='Z = 0')
#ax.legend(handles=[line_proxy], loc='upper right')

xmax = np.max(np.abs(x_full))
ymax = np.max(np.abs(y))

ax.set_xlabel('x', fontweight='bold', fontsize=axisFontSize)
ax.set_ylabel('y', fontweight='bold', fontsize=axisFontSize)
ax.set_xlim(-xpad * xmax, xpad * xmax)
ax.set_ylim(-(ymax + ypad), ymax + ypad)
title_save = "varphi_contour_upright"
ax.set_title(r'$\varphi(x,y)$', fontweight='bold', fontsize=titleFontSize)
ax.set_aspect('equal')  # remove if you don’t want square data aspect
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig(title_save + '.png', dpi=400)
#plt.show()


Rs = 0.00475

#####------------------------------------------------------------------------------------------SIDEWAYS
# ========================== ROTATED 90° VERSION ==========================
# Same plot as your block, but rotated: swap axes (X↔Y) and use Z_full.T.
# Keeps your legend-on-right with φ levels and the φ_min label above the legend.

# --- build rotated coords/field ---
Xr = Rs*Y.T*100                  # horizontal axis after rotation
Yr = Rs*X.T*100                  # vertical   axis after rotation
Zr = Z_full.T                  # transpose so values stay with the right coords

# (optional) mask non-finite so contouring skips them
Zr = np.where(np.isfinite(Zr), Zr, np.nan)
Z  = np.where(Zr < 0, Zr, np.nan)     # keep your original filtering logic if desired
vmin = np.nanmin(Zr)                  # for φ_min in the legend/title

fig, ax = plt.subplots(figsize=(12, 8))

# filled contours + line contours (same styling as yours)
cf = ax.contourf(Xr, Yr, Zr, levels=nLevs, cmap='viridis')    # change levels/cmap as you like
cs = ax.contour (Xr, Yr, Z,  levels=levels, colors=colors, linewidths=1.0)

levels = np.atleast_1d(cs.levels)
handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(nLevs)]
labels  = [fr'$\varphi={lev:.2f}$' for lev in levels]

# ----- legend on the right -----
leg = ax.legend(handles, labels, title='Levels',
                loc='center left', bbox_to_anchor=(1.02, 0.5),
                frameon=False, borderaxespad=0)

# φ_min above the legend (like a header)
fig.text(0.94, 0.75, rf'$\varphi_{{\min}}={vmin:.3g}$',
         ha='center', va='top', fontsize=titleFontSize-4)

plt.tight_layout(rect=[0, 0, 0.82, 1])

# --- optional: add φ=0 contour in bold red (uses rotated arrays) ---
Z_for_zero = np.where(np.isfinite(Zr), Zr, np.nan)
c0 = ax.contour(Xr, Yr, Z_for_zero, levels=[0.0], colors='crimson',linewidths=1.0, linestyles='-', zorder=10)

ax.set_xlabel('z [cm]', fontsize=axisFontSize, fontweight='bold')
ax.set_ylabel('r [cm]', fontsize=axisFontSize, fontweight='bold')
ax.set_xlim(-(ypad + ymax)*Rs*100, (ypad + ymax)*Rs*100)
ax.set_ylim(-(xmax * xpad)*Rs*100, (xmax * xpad)*Rs*100)

title_save = 'varphi_contour_sideways'
ax.set_title(r'$\varphi(x,y)$', fontsize=titleFontSize, fontweight='bold')
ax.set_aspect('equal')                      # remove if you don’t want square data aspect
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig(title_save + '.png', dpi=dpi_res)
#plt.show()
# ======================================================================






#####--------------------------------------------------------------------------------------3D CONTOUR
# ===== 3D SURFACE + CONTOUR =====
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
Z = Z.T
fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))

# --- enforce 1:1 in the XY plane, and scale Z by a factor `ze` ---
ze = 0.2  # vertical scale (try 2–8)

# Centered equal limits in XY so they render 1:1
x0, x1 = np.nanmin(X), np.nanmax(X)
y0, y1 = np.nanmin(Y), np.nanmax(Y)
xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
r = max(rx, ry)                       # same half-range on x and y

ax3.set_xlim(xm - r, xm + r)
ax3.set_ylim(ym - r, ym + r)

# Make the axes box have ratio 1:1:ze (this scales z visually only)
ax3.set_box_aspect((1, 1, ze))


# Surface
surf = ax3.plot_surface(
    X, Y, Z,
    cmap='viridis',
    linewidth=0,
    antialiased=True,
    alpha=0.95
)

# Contour lines on the surface
ax3.contour3D(
    X, Y, Z,
    levels=12,
    colors='k',
    linewidths=0.6
)

# Optional: project contours onto the bottom z-plane
zmin = np.nanmin(Z)
ax3.contour(
    X, Y, Z,
    zdir='z', offset=zmin,
    levels=12,
    cmap='viridis',
    alpha=0.7
)

# Axes labels / limits / aspect
ax3.set_xlabel('x', fontsize=axisFontSize, fontweight='bold', labelpad=8)
ax3.set_ylabel('y', fontsize=axisFontSize, fontweight='bold', labelpad=8)
ax3.set_zlabel(r'$\varphi$', fontsize=axisFontSize, fontweight='bold', labelpad=6)



# View & grid
ax3.view_init(elev=30, azim=-60)     # change angles as you like
ax3.grid(True)

# Colorbar
cbar3 = fig3.colorbar(surf, ax=ax3, pad=0.12, shrink=0.7)
cbar3.set_label(r'$\varphi$')

plt.tight_layout()
#plt.show()



#####=============================================================================================#####
#####============================================REVIVED===============================================
#####=============================================================================================#####
del r, x, y, X, Y

###-----Set domain
Rw = 6.1e-3                 #Flux-conserving (aka conducting) wall radius [m]
Xs = 0.75                   #Normalized separatrix radius [\]
Rs = Xs * Rw                #Separatrix radius [m]
Zs = E * Rs                 #Separatrix half-length [m]
Rp = Rs/2                   #Plasma major radius [m]
a = Rs - Rp                 #Plasma minor radius [m]

h = 50e-3                   #domain half-height [m]
zmax = h
rmax = Rw*2
Nr = 99
Nz = 199
dr = Rw/Nr
dz = 2*h/Nz
r = np.arange(0, rmax+dr, dr)
z = np.arange(-zmax, zmax+dz, dz)
R, Z = np.meshgrid(r,z)


###-----Solving for the Actual Flux
psi = np.zeros((len(r), len(z)))

psi0 = 1
for i in range(len(r)):
    for j in range(len(z)):
        x = r[i]
        y = z[j]
        psi[i,j] = varphi_sol(x, y, A, c_h) * psi0
        

#print(psi)
#####----------------------------------------------------------------------------------------PLOTTING
orientation = "upright" #"upright" or "sideways"

# Reflect across the y-axis (x -> -x) and stitch the two halves.
if r[0] == 0:
    r_full = np.concatenate((-r[::-1], r[1:]))             # avoid duplicate 0
    P_full = np.concatenate((psi[::-1, :], psi[1:, :]), axis=0)
else:
    r_full = np.concatenate((-r[::-1], r))
    P_full = np.concatenate((psi[::-1, :], psi), axis=0)
R, Z = np.meshgrid(r_full, z, indexing='ij') 

# (Optional) mask non-finite values so plotting skips them
P = np.where(np.isfinite(P_full), P_full, np.nan)
P = np.where(P < 0, P, np.nan)  # (keep your original filtering logic if needed)
vmin = np.nanmin(P_full)   # or whatever array you contour

# --- build rotated coords/field ---
if orientation=="sideways":
    Xr = Z.T                  # horizontal axis after rotation
    Yr = R.T                  # vertical   axis after rotation
    Pr = P_full.T
elif orientation=="upright":
    Xr = R                  # horizontal axis after rotation
    Yr = Z                  # vertical   axis after rotation
    Pr = P_full                  # transpose so values stay with the right coords

# (optional) mask non-finite so contouring skips them
Pr = np.where(np.isfinite(Pr), Pr, np.nan)
P  = np.where(Pr < 0, Pr, np.nan)     # keep your original filtering logic if desired
vmin = np.nanmin(Pr)                  # for φ_min in the legend/title

fig, ax = plt.subplots(figsize=(12, 8))

# filled contours + line contours (same styling as yours)
#cf = ax.contourf(Xr, Yr, Pr, levels=nLevs, cmap='viridis')    # change levels/cmap as you like
cs = ax.contour (Xr, Yr, P,  levels=levels, colors=colors, linewidths=1.0)

levels = np.atleast_1d(cs.levels)
handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(nLevs)]
labels  = [fr'$\varphi={lev:.2f}$' for lev in levels]

# ----- legend on the right -----
leg = ax.legend(handles, labels, title='Levels',
                loc='center left', bbox_to_anchor=(1.02, 0.5),
                frameon=False, borderaxespad=0)

# φ_min above the legend (like a header)
fig.text(0.74, 0.75, rf'$\psi_{{\min}}={vmin:.3g}$',
         ha='center', va='top', fontsize=titleFontSize-4)

plt.tight_layout(rect=[0, 0, 0.82, 1])

# --- optional: add φ=0 contour in bold red (uses rotated arrays) ---
P_for_zero = np.where(np.isfinite(Pr), Pr, np.nan)
c0 = ax.contour(Xr, Yr, P_for_zero, levels=[0.0], colors='crimson',linewidths=1.0, linestyles='-', zorder=10)

ax.set_xlabel('z [m]', fontsize=axisFontSize, fontweight='bold')
ax.set_ylabel('r [m]', fontsize=axisFontSize, fontweight='bold')
#ax.set_xlim(-(ypad + ymax)*Rs*100, (ypad + ymax)*Rs*100)
#ax.set_ylim(-(xmax * xpad)*Rs*100, (xmax * xpad)*Rs*100)

title_save = 'psi_contour_sideways'
ax.set_title(r'$\psi(x,y)$', fontsize=titleFontSize, fontweight='bold')
ax.set_aspect('equal')                      # remove if you don’t want square data aspect
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig(title_save + '.png', dpi=dpi_res)
print(title_save, " printed and saved!")
#plt.show()







#################################################
#####               PRINTING                #####
#################################################
t1code02 = time.perf_counter()
##########################################################################
#####                           OUTPUT                               #####
##########################################################################
print("\n\n")
print("FRC_02: Grad-Shafranov Equilibrium Derivation =================")

print("\n \t=== FRC Model Parameters ===\n")
print(f"\t {'Plasma Major Radius (Rp = Rs/2) [m]:':40s} {Rp:.3e}")
print(f"\t {'Plasma Minor Radius (a = Rs-Rp) [m]:':40s} {a:.3e}")
print(f"\t {'Inverse aspect ratio (ε = a/Rp):':40s} {eps:.3f}")
print(f"\t {'Triangularity (δ):':40s} {delta:.3f}")
print(f"\t {'Solov\'ev constant (A):':40s} {A:.3f}")
print(f"\t {'Separatrix shift (σ_sep):':40s} {sig_sep:.3f}")
print(f"\t {'Triangularity angle (α) [rad]:':40s} {alpha:.3f}")

print()
print(f"\t {'Conducting wall radius (Rw) [m]:':40s} {Rw:.3e}")
print(f"\t {'Normalized separatrix radius (Xs):':40s} {Xs:.3f}")
print(f"\t {'Separatrix radius (Rs) [m]:':40s} {Rs:.3e}")
print(f"\t {'Separatrix half-length (Zs) [m]:':40s} {E*Rs:.3e}")
print(f"\t {'Elongation:':40s} {E:.3f}")

print()
print(f"\t {'Curvature constant N1:':40s} {N1:.3f}")
print(f"\t {'Curvature constant N3:':40s} {N3:.3f}")

print("\n \t============================\n")

print_tabbed("Phi_h Matrix", Phi_h)
print_tabbed("Phi BC Matrix", Phi_bc)
print_tabbed("varphi_p Column Vector", varphi_P)
print_tabbed("varphi_p_BC Column Vector", varphi_P_bc)

print("\t Constants")
for i in range(len(c_h)):
    print(f"\t    c{i+1} = ", c_h[i])


#print("\t --------------------------------")
#print("\t Phi_bc - Phi_h")
#print("\t", Phi_bc - Phi_h)

#print("\t varphi_p - varphi_p_bc")
#print("\t", varphi_P - varphi_P_bc)

print("\t frc02 Compute Time = {:.3e} [s]".format(t1code02-t0code02))









sys.exit()
# Reflect across the y-axis (x -> -x) and stitch the two halves.
if x[0] == 0:
    x_full = np.concatenate((-x[::-1], x[1:]))             # avoid duplicate 0
    Z_full = np.concatenate((psi[::-1, :], psi[1:, :]), axis=0)
else:
    x_full = np.concatenate((-x[::-1], x))
    Z_full = np.concatenate((psi[::-1, :], psi), axis=0)
X, Y = np.meshgrid(x_full, y, indexing='ij')  # X,Y shapes (2*Nx-(x[0]==0), Ny)

# (Optional) mask non-finite values so plotting skips them
Z = np.where(np.isfinite(Z_full), Z_full, np.nan)
Z = np.where(Z < 0, Z, np.nan)  # (keep your original filtering logic if needed)
vmin = np.nanmin(Z_full)   # or whatever array you contour

fig, ax = plt.subplots(figsize=(7, 8))
cf = ax.contourf(X, Y, Z, levels=nLevs, cmap='viridis')  # change levels/cmap as you like
cs = ax.contour(X, Y, Z, levels=nLevs, colors=colors, linewidths=1.0)

levels = np.atleast_1d(cs.levels)
handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(nLevs)]
labels  = [fr'$\psi={lev:.2f}$' for lev in levels]

ax.legend(handles, labels, title=r'Levels',
          loc='center left', bbox_to_anchor=(1.02, 0.5),
          frameon=False, borderaxespad=0.0)
fig.text(0.8, 0.7, rf'$\psi_{{\min}}={vmin:.3g}$',
         ha='center', va='top', fontsize=titleFontSize-4)
plt.tight_layout(rect=[0, 0, 0.82, 1])

# --- Add Z = 0 contour line in bold red (no .collections access) ----
Z_for_zero = np.where(np.isfinite(Z_full), Z_full, np.nan)

#line_proxy = Line2D([0], [0], color='crimson', lw=3, label='Z = 0')
#ax.legend(handles=[line_proxy], loc='upper right')

xmax = np.max(np.abs(x_full))
ymax = np.max(np.abs(y))

ax.set_xlabel('r', fontweight='bold', fontsize=axisFontSize)
ax.set_ylabel('z', fontweight='bold', fontsize=axisFontSize)
#ax.set_xlim(-xpad * xmax, xpad * xmax)
#ax.set_ylim(-(ymax + ypad), ymax + ypad)
title_save = "psi_contour_upright"
ax.set_title(r'$\psi(x,y)$', fontweight='bold', fontsize=titleFontSize)
ax.set_aspect('equal')  # remove if you don’t want square data aspect
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig(title_save + '.png', dpi=400)
#plt.show()




sys.exit()







































#################################################
#####               DEBUGGING               #####
#################################################

###-----POINTS OF ANALYSIS
xx = 0.5        #points of analysis
yy = -3.74321
c2 = c_h[1]
c4 = c_h[3]
c6 = c_h[5]
varphi_hand = (xx**4 / 8) + c2*xx**2 + c4*(xx**4 - 4*(xx*yy)**2) + c6*(xx**6-12*xx**4*yy**2 + 8*xx**2*yy**4)
varphi_comp = varphi_sol(xx, yy, A, c_h)

diffx = x-xx
diffy = y-yy
xid = np.argmin(np.abs(x-xx))
yid = np.argmin(np.abs(y-yy))

print("Points of Analysis")
print("x = ", xx, "   y = ", yy)
print("xid = ", int(xid), "   ", "yid = ", yid)

print("varphi_hand = ", varphi_hand)
print("varphi_comp = ", varphi_comp)
print("varphi_indx = ", varphi[xid, yid])




###-----SLICING
x_slice = np.arange(0, 2.2, 0.2)
y_slice = np.arange(0, 10, 1)
for k in range(len(y_slice)):
    varphi_mid = np.zeros_like(x)
    for i in range(len(x)):
        #varphi_mid[i] = varphi_sol(x_slice[k], y[i], 0, c_h)
        varphi_mid[i] = varphi_sol(x[i], y_slice[k], 0, c_h)

    plt.plot(x, varphi_mid)
    plt.grid('both')
    plt.xlabel('x')
    plt.ylabel('$\\varphi$')
    plt.ylim(-0.75, 0.1)
    plt.title('Radial Contour Slice at y = {:.2f}'.format(y_slice[k]))
    plt.savefig('radial_flux_y{:.2f}.png'.format(y_slice[k]), dpi=400)
    plt.close()




###-----SCATTER PLOT
varphi_neg = np.where(np.isfinite(varphi) & (varphi<= 0), varphi, np.nan)
#plt.plot(x, y, varphi_neg)
#plt.close()

fig2 = plt.figure(figsize=(8, 8))
ax2  = fig2.add_subplot(111, projection='3d')

#ax2.scatter(X, Y, varphi_neg, marker='o', color='blue')
for i in range(len(x)):
    for j in range(len(y)):
        xx = x[i]
        yy = y[j]
        varphi_comp = varphi_sol(xx, yy, A, c_h)
        ax2.scatter(xx, yy, varphi_comp, marker='o', color='red')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_aspect('equal', adjustable='box')
#plt.show()








###--------------------_NEW PLOT
# Build the (x,y) grid with the same layout as your i/j loops
X, Y = np.meshgrid(x, y, indexing='ij')   # shape: (len(x), len(y))

# Evaluate varphi on the grid (vectorized if possible)
try:
    # If varphi_sol supports numpy arrays, this is fastest
    varphi = varphi_sol(X, Y, A, c_h)
except Exception:
    # Otherwise, safely vectorize
    varphi = np.vectorize(lambda xi, yj: varphi_sol(xi, yj, A, c_h),
                          otypes=[float])(X, Y)

# Keep only finite values; turn ±inf into NaN
varphi = np.where(np.isfinite(varphi), varphi, np.nan)

# Positive-only version (handy for plotting only positives)
varphi_pos = np.where(varphi > 0, varphi, np.nan)

# Diagnostics
num_nans   = np.isnan(varphi).sum()
num_pos    = np.isfinite(varphi_pos).sum()
print(f"varphi shape: {varphi.shape} | NaNs: {num_nans} | positive cells: {num_pos}")

# (Optional) peek at values
# print(varphi)



sys.exit()

###########################
bgcol = "white"     #background color
lncol = "black"     #line color
fig, ax = plt.subplots(figsize=(5,8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)

varphi_neg = np.where(np.isfinite(varphi) & (varphi<= 0), varphi, np.nan)
# Contour plot
cs = ax.contour(Y, X, varphi_neg, levels=12,
                cmap="rainbow_r", linewidths=1.25)

# Colorbar with position control
cbar = fig.colorbar(cs, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
cbar.set_label(r'$\varphi_{\mathrm{sol}}$', color=bgcol, fontsize=12, fontweight='bold')
# Put φ(x,y) on top of colorbar
cbar.ax.set_title(r'$\varphi(x,y)$', fontsize=12, fontweight='bold', color='black')


# Axis labels + limits
xc = 1.15
yc = 1.15
ax.set_xlabel(r'$x = r/R_s$', color=lncol, fontweight='bold')
ax.set_ylabel(r'$y = z/R_s$', color=lncol, fontweight='bold')
ax.set_xlim([-2*xc, 2*xc])
ax.set_ylim([-E*yc, E*yc])
ax.set_aspect('equal', adjustable='box')
ax.set_title(r'$\varphi(x,y)$', color=lncol, fontweight='bold')
#ax.grid(True, which='both', color=lncol, alpha=0.2, linewidth=0.5, linestyle='--')

# Set black background
fig.patch.set_facecolor(bgcol)   # outside area
ax.set_facecolor(bgcol)          # plot area
cbar.ax.set_facecolor(bgcol)     # colorbar background

# Make ticks/labels white
ax.tick_params(colors=lncol)

# --- Make ticks/labels bold ---
ax.tick_params(colors=lncol, labelsize=10, width=1.2)
plt.setp(ax.get_xticklabels(), color=lncol, fontweight='bold')
plt.setp(ax.get_yticklabels(), color=lncol, fontweight='bold')
plt.setp(cbar.ax.get_yticklabels(), color=lncol, fontweight='bold')
plt.grid('both')
plt.savefig('contour2.png', dpi=400)
plt.close()









t1code = time.perf_counter()
print("Completed in: ", t1code-t0code, "s")
#################################################
#####               PLOTTING                #####
#################################################


bgcol = "white"     #background color
lncol = "black"     #line color

# 1) mirror across x=0  (x is along axis=0 if you used meshgrid(..., indexing="ij"))
#varphi_full = np.concatenate([np.flip(varphi, axis=1), varphi], axis=1)
#X_full      = np.concatenate([-np.flip(X, axis=1),    X     ], axis=1)
#Y_full      = np.concatenate([ np.flip(Y, axis=1),    Y     ], axis=1)

# assume x is symmetric -> extend to negative-x by flipping
varphi_full = np.hstack([np.flip(varphi, axis=1), varphi])  # concatenate flipped and original
X_full = np.hstack([-np.flip(X, axis=1), X])
Y_full = np.hstack([np.flip(Y, axis=1), Y])



# 2) mask AFTER mirroring: keep only positive values
varphi_pos = np.where(np.isfinite(varphi_full) & (varphi_full >= 1), varphi_full, np.nan)
#varphi_pos = np.where(np.isfinite(varphi) & (varphi >= 0), varphi, np.nan)

# 3) plot


fig, ax = plt.subplots(figsize=(5,8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)

# Contour plot
#cs = ax.contour(X, Y, -varphi_pos, levels=12,
#                cmap="rainbow_r", linewidths=1.25)
cs = ax.contour(X_full, Y_full, varphi_full, levels=10,
                cmap="rainbow_r", linewidths=1.25)

# Colorbar with position control
cbar = fig.colorbar(cs, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
cbar.set_label(r'$\varphi_{\mathrm{sol}}$', color=bgcol, fontsize=12, fontweight='bold')
# Put φ(x,y) on top of colorbar
cbar.ax.set_title(r'$\varphi(x,y)$', fontsize=12, fontweight='bold', color='black')


# Axis labels + limits
xc = 1.15
yc = 1.15
ax.set_xlabel(r'$x = r/R_s$', color=lncol, fontweight='bold')
ax.set_ylabel(r'$y = z/R_s$', color=lncol, fontweight='bold')
ax.set_xlim([-2*xc, 2*xc])
ax.set_ylim([-E*yc, E*yc])
ax.set_aspect('equal', adjustable='box')
ax.set_title(r'$\varphi(x,y)$', color=lncol, fontweight='bold')
#ax.grid(True, which='both', color=lncol, alpha=0.2, linewidth=0.5, linestyle='--')

# Set black background
fig.patch.set_facecolor(bgcol)   # outside area
ax.set_facecolor(bgcol)          # plot area
cbar.ax.set_facecolor(bgcol)     # colorbar background

# Make ticks/labels white
ax.tick_params(colors=lncol)

# --- Make ticks/labels bold ---
ax.tick_params(colors=lncol, labelsize=10, width=1.2)
plt.setp(ax.get_xticklabels(), color=lncol, fontweight='bold')
plt.setp(ax.get_yticklabels(), color=lncol, fontweight='bold')
plt.setp(cbar.ax.get_yticklabels(), color=lncol, fontweight='bold')



title_save = 'Scaled Flux'
# --- Print c_h values on the left, vertically aligned ---
text_lines = [
    f"c1 = {c_h[0]:.3e}",
    f"c2 = {c_h[1]:.3e}",
    f"c3 = {c_h[2]:.3e}",
    f"c4 = {c_h[3]:.3e}",
    f"c5 = {c_h[4]:.3e}",
    f"c6 = {c_h[5]:.3e}",
    f"c7 = {c_h[6]:.3e}",
    f"c8 = {c_h[7]:.3e}",
    f"c9 = {c_h[8]:.3e}",
    f"c10 = {c_h[9]:.3e}",
    f"c11 = {c_h[10]:.3e}",
    f"c12 = {c_h[11]:.3e}"
]

# start Y ~ middle and offset for each line
y0 = 0.9  
dy = 0.05  
for i, line in enumerate(text_lines):
    ax.text(-1.5, y0 - i*dy, line, transform=ax.transAxes,
            ha='left', va='center', fontsize=10, color='black', fontweight='bold')

plt.title(title_save)
plt.close()
































sys.exit()

# 3) plot
bgcol = "white"
lncol = "black"
xc = 1.15
yc = 1.15
fig, ax = plt.subplots(figsize=(5,8))
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1)

# Contour plot
#cs = ax.contour(X, Y, varphi_pos, levels=12,
#                cmap="rainbow_r", linewidths=1.25)
cs = ax.contour(X, Y, varphi, levels=10,
                cmap="rainbow_r", linewidths=1.25)

# Colorbar with position control
cbar = fig.colorbar(cs, ax=ax, orientation="vertical", shrink=0.8, pad=0.05)
cbar.set_label(r'$\varphi_{\mathrm{sol}}$', color=bgcol)

# Axis labels + limits
ax.set_xlabel(r'$x = r/R_s$', color=lncol)
ax.set_ylabel(r'$y = z/R_s$', color=lncol)
ax.set_xlim([-2*xc, 2*xc])
ax.set_ylim([-E*yc, E*yc])
ax.set_aspect('equal', adjustable='box')
ax.set_title(r'$\varphi(x,y)$', color=lncol)
#ax.grid(True, which='both', color=lncol, alpha=0.2, linewidth=0.5, linestyle='--')

# Set black background
fig.patch.set_facecolor(bgcol)   # outside area
ax.set_facecolor(bgcol)          # plot area
cbar.ax.set_facecolor(bgcol)     # colorbar background

# Make ticks/labels white
ax.tick_params(colors=lncol)
cbar.ax.yaxis.set_tick_params(color=lncol)
plt.setp(cbar.ax.get_yticklabels(), color=lncol)
title_save = 'Scaled Flux'
plt.savefig(title_save, dpi=dpi_res)
#plt.show()











