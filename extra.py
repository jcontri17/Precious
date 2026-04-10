#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Elongation axis (matches your plot)
# -----------------------------
kappa = np.linspace(1.0, 10.0, 400)   # elongation = Zs/Rs

# -----------------------------
# Choose how geometry/params vary with elongation
# (THIS is what controls the curve shapes)
# -----------------------------
def Rs_of_kappa(k):
    """
    Separatrix radius Rs [m] vs elongation.
    If you want something close to your screenshot shapes, you typically need
    some nontrivial k-dependence here.
    """
    Rs0 = 0.006  # 0.6 cm baseline
    # mild decrease with kappa (edit this if needed)
    return Rs0 * (k**(-0.15))

def vA_of_kappa(k):
    """Alfvén speed vA [m/s]."""
    vA0 = 2.0e5
    return vA0 * (k**0.05)

def eta_perp_of_kappa(k):
    """Perpendicular resistivity eta_perp [Ohm*m]."""
    eta0 = 2.0e-6
    return eta0 * (k**0.4)

def Xs_of_kappa(k):
    """Xs = Rs/Rw (dimensionless)."""
    Rw = 0.015  # 1.5 cm wall radius (example)
    return Rs_of_kappa(k) / Rw

def rho_ie_of_kappa(k):
    """rho_{i,e} [m] (put in whatever your model uses)."""
    rho0 = 2.0e-4
    return rho0 * (k**0.2)

def n_of_kappa(k):
    """Number density n [m^-3]."""
    n0 = 2.0e21
    return n0 * (k**(-0.25))

def T_of_kappa(k):
    """Temperature T [J]. (If you prefer eV, convert before sqrt(kB*T))."""
    # Example: 200 eV
    eV = 1.602176634e-19
    T0_eV = 200.0
    T0 = T0_eV * eV
    return T0 * (k**0.1)

# -----------------------------
# Constants
# -----------------------------
mu0 = 4e-7 * np.pi
kB  = 1.380649e-23

# A_Brem is a model constant in your slide formula (units depend on how you define n, T)
# Treat as a tunable constant to match your normalization.
A_Brem = 5.0e-37

# -----------------------------
# Derived quantities
# -----------------------------
Rs = Rs_of_kappa(kappa)
Zs = kappa * Rs
vA = vA_of_kappa(kappa)

tau_MHD   = Zs / vA
tau_class = mu0 * Rs**2 / (16.0 * eta_perp_of_kappa(kappa))

# LSX formula from your slide:
# tau_phi,LSX = 6.5e-5 * sqrt(Xs) * (Rs / sqrt(rho_{i,e}))^2.14
Xs = Xs_of_kappa(kappa)
rho_ie = rho_ie_of_kappa(kappa)
tau_LSX = 6.5e-5 * np.sqrt(Xs) * (Rs / np.sqrt(rho_ie))**2.14

# Brem formula from your slide:
# tau_phi,Brem = 2*sqrt(kB*T)/(A_Brem*n)
n = n_of_kappa(kappa)
T = T_of_kappa(kappa)
tau_Brem = 2.0 * np.sqrt(kB * T) / (A_Brem * n)

# -----------------------------
# Plot (styled like your screenshot)
# -----------------------------
y1 = tau_class / tau_MHD
y2 = tau_LSX   / tau_MHD
y3 = tau_Brem  / (10.0 * tau_MHD)

fig, ax = plt.subplots(figsize=(9.6, 5.4))
ax.plot(kappa, y1, linewidth=3, label=r'$\tau_{class}\ /\ \tau_{MHD}$')
ax.plot(kappa, y2, linewidth=3, label=r'$\tau_{LSX}\ /\ \tau_{MHD}$')
ax.plot(kappa, y3, linewidth=3, label=r'$\tau_{Brem}\ / 10\tau_{MHD}$')

ax.set_xlim(1, 10)
ax.set_xlabel("Elongation", fontsize=16, fontweight='bold')
ax.set_ylabel(r'$\tau_{\phi}\ /\ \tau_{MHD}$', fontsize=16, fontweight='bold')

ax.grid(True, which='both', alpha=0.6)
ax.legend(loc='upper right', framealpha=1.0, edgecolor='k')

plt.tight_layout()
plt.show()

