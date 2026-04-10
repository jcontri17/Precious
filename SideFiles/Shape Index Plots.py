#######################################################################################################
#######################################################################################################
#################################                  IMPORTS                 ############################
#######################################################################################################
#######################################################################################################

import matplotlib.pyplot as plt #library for plotting 
from matplotlib.animation import FuncAnimation
from sympy import *
import numpy as np              #library for basic math functions 
import cmath                    #library for complex numbers
import sys                      #library for various system-specific parameters and functions
import time                     #library for time-based functions
import array 

from scipy.optimize import fsolve
from scipy.optimize import root 
from mpl_toolkits.mplot3d import Axes3D                         #registers the 3D projection 
from matplotlib.ticker import AutoMinorLocator
from matplotlib.path import Path
from scipy.interpolate import RegularGridInterpolator 
from matplotlib.colors import LogNorm

from plottingParameters import *  

#######################################################################################################
#######################################################################################################
#################################                  MAIN                    ############################
#######################################################################################################
#######################################################################################################
Lconv = 1e-2
###---------------------------------------------------------------------------------CONSTANT PARAMETERS 
Rw = 0.61*Lconv        #wall radius [m]
Rc = Rw                 #coil radius [m]
Xs = 0.75               #normalized separatrix ratio [\]
Rs = Xs * Rw            #separatrix radius [m]

h = 4.00*Lconv         #length of liner [m]
Lx = 3.00*Lconv        #FRC length [m]
Z0 = Lx / 2             #FRC half-length [m]
A = Rs                  #FRC semi-minor axis [m]
B = Z0                  #FRC semi-major axis [m]

sig = 1.5               #flare parameter; adjustable parameter that's fixed for Steinhauer's paper
f = 1.5                 #internal psi error factor for Sporer's approximation
eps = A / B             #inverse elongation
Bw = 10                 #Magnetic field at the midplane at the wall [T] from Steinhaur
B00 = Bw                #Sporer vacuum field [T] still working out how this relates to Steinhauer

acc = 100               #number of variables in list -- accuracy 

###---------------------------------------------------------------------------------FUNCTIONS
def external_E_system(Es, eps, Xs, sig):        #external parameters
    E0, E1, E2, alpha = Es
    eq1 = E0 - E1 * (4/eps**2) + E2 * (2*eps**2*alpha/(1-alpha**2)**2)
    eq2 = E0 + E1 + 2*alpha*E2 / np.sqrt(eps**2+alpha**2)
    e0 = eps**2*Xs**3*E2/4/sig**2
    e1 = (alpha+sig)/ np.sqrt(eps**2+Xs**2*(alpha+sig)**2)
    e2 = (alpha-sig)/ np.sqrt(eps**2+Xs**2*(alpha-sig)**2)
    e3 = 2*alpha/ np.sqrt(eps**2+(Xs*alpha)**2)
    eq3 = E1 - e0 * (e1 + e2 - e3)
    eq4 = E0 + 2*E1/Xs**2 - eps**2*alpha*Xs**3*E2 / ((alpha*Xs)**2+eps**2)*1.5 - 1
    return [eq1, eq2, eq3, eq4]

def Rs_curavture(a,b,z):                        #derived from flux equation 
    r_prime = -((2**(3/2))*(z**3)*(a)) / ((b**2)*np.sqrt(3*(a**2)*(b**2) + (2*b**4) -2*(z**4)))
    r_2prime = ((2**(3/2))*a*(z**2)*(2*(z**4)-6*(b**4)-9*(a**2)*(b**2))) / ((b**2)*(-2*(z**4) + 2*(b**4)+3*(a**2)*(b**2)))**(3/2)
    result = (((r_2prime)**(2))**(1/2)) / ((r_prime)**(2)+1)**(3/2)
    return result 

### ---- SHAPE INDEX FUNCTIONS 
def shape_index_N1(a, b, Rs):                    #improved analysis of equilbria --- equation 7
    N = b**2 / a / Rs
    return N

def shape_index_N2(eps, E0, E1, E2, alpha):     #improved analysis of equilbria --- equation 7
    lamb = (1 + (eps/alpha)**2)**(-0.5)
    N1 = 4*eps*E1
    N2 = 3*alpha*lamb**5*E2
    D1 = -eps*E1 
    D2 = alpha*lamb * E2 *(2+lamb**2)
    N = (N1+N2)/(eps**2*(D1+D2))
    return N

def shape_index_N3(a, b, Xs):                   #improved analysis of equilbria --- equation 8 
    N = (1+(0.8+Xs**2)*(b/a-1)**2)**(-1)
    return N

### ---- ANIMATION FUNCTIONS
def separatrix_shape_pos(t, a, b):      #let z = t 
    result = (a*((3/2)*(a**2)*(b**2)+(b**4)-(t**4))**(0.5))/(b**2)
    return result 

def separatrix_domain(a,b):             #domain for x-axis (where x=z=t)
    result = ((3/2)*(a**2)*(b**2) + b**4)**(1/4)
    return result 

def separatrix_range(a,b): 
    result = ((3/2)*a**2*b**2 + b**4)**(0.25)
    return result 

#######################################################################################################
#######################################################################################################
#################################                 PARAMETERS               ############################
#######################################################################################################
####################################################################################################### 

#elongation (b/a) data points between 1 and 10
elongation_list = np.linspace(1,10,num = acc)

#eps values (inverse elongation)
eps_listValues = 1/elongation_list
eps_listValues_copy = eps_listValues.copy()
#a values and corresponding b values as determined by elongation 
b_list = np.linspace(0.001, 0.02, num = acc)
a_list = b_list / elongation_list


#X_s values between 0.3 and 0.9: 
Xs_listValues = np.linspace(0.3,0.9,num=5)
"""
#Determining the external parameters from the various eps & Xs values, and constant sig
e_params = []

for eps in eps_listValues_copy:
    for Xs in Xs_listValues:
        #Initial guess for [E0, E1, E2, alpha] ---> Subject to change to be in terms of something else 
        initial_guess = [1.0, 0.5, 0.3, 0.2]
        
        #solve the system of equations 
        result = root(external_E_system, initial_guess, args= (eps, Xs, 1.5))
    #ensuring that correct results get added to the parameter list
    if result.success: 
        E0, E1, E2, a = result.x
        e_params.append(
            {'eps': eps, 
             'Xs': Xs,
             'E0': E0,
             'E1': E1, 
             'E2': E2,
             'alpha': a
             }) 
    else: 
        #print(f"Solution failed for eps={eps}, Xs={Xs}")
        dummy1 = np.where(eps_listValues_copy == eps)
        eps_listValues_copy[dummy1] = 0

for index, eps in enumerate(eps_listValues): 
    if eps == eps_listValues_copy[index]:
        pass
    else: 
        a_list[index] = 0
        b_list[index] = 0
        elongation_list[index] = 0
        
eps_listValues = eps_listValues_copy

a_list = a_list[a_list != 0]
b_list = b_list[b_list != 0]
elongation_list = elongation_list[elongation_list != 0]
eps_listValues = eps_listValues[eps_listValues != 0]

"""
"""
#######################################################################################################
#######################################################################################################
#################################                 PLOTS                    ############################
#######################################################################################################
#######################################################################################################                                                            
###-------------------------------------------------------------------------------ylist; N_1 (space index) 

ylist_N1 = shape_index_N1(a_list, b_list, Rs_7_list)

    #N1 plot
plt.figure(num=0, dpi=dpi_res)
plt.grid()
plt.plot(elongation_list, ylist_N1)
plt.title("elongation [b/a] vs Shape Index [N1]", fontsize = 16)
plt.xlabel("elongation [b/a]", fontsize = 12)  # Label for the x-axis
plt.ylabel("Shape Index [N1]", fontsize = 12)  # Label for the y-axis
title_save = "elongation vs Shape Index N1"
plt.savefig(title_save + ".png", dpi = dpi_res)

###-------------------------------------------------------------------------------ylist; N_2 (space index) 
ylist_N2 = []
xlist_N2 = []
index = 0
for i in eps_listValues:
    ylist_N2.append(shape_index_N2(i, e_params[index]['E0'],e_params[index]['E1'],e_params[index]['E2'], e_params[index]['alpha']))
    xlist_N2.append(1/i)
    index +=1
   
    #N2 plot 
plt.figure(num=1, dpi= dpi_res)
plt.grid()
plt.plot(xlist_N2, ylist_N2)
plt.title("elongation [b/a] vs Shape Index [N2]", fontsize = 16)
plt.xlabel("elongation [b/a]", fontsize =12)  # Label for the x-axis
plt.ylabel("Shape Index [N2]", fontsize = 12)  # Label for the y-axis
title_save = "elongation vs Shape Index N2"
plt.savefig(title_save + ".png", dpi = dpi_res)
"""
###-------------------------------------------------------------------------------ylist; N_3 (space index) 
xlist_N3 = elongation_list
all_ylist_N3 = []
for i in Xs_listValues:
    ylist_N3_Xs = shape_index_N3(a_list, xlist_N3*a_list, i)
    all_ylist_N3.append(ylist_N3_Xs)
    
"""
    #N3 plot
index = 0
plt.figure(num=2, dpi=dpi_res)
for i in all_ylist_N3:
    lab_el = "{:.2f}".format(Xs_listValues[index])
    plt.plot(xlist_N3, i, label = lab_el )
    index += 1
plt.grid()
plt.title("elongation [b/a] vs Shape Index [N3]", fontsize = 16)
plt.xlabel("elongation [b/a]", fontsize =12)  # Label for the x-axis
plt.ylabel("Shape Index [N3]", fontsize = 12)  # Label for the y-axis
plt.legend()
title_save = "elongation vs Shape Index N3"
plt.savefig(title_save + ".png", dpi = dpi_res)
"""
#######################################################################################################
#######################################################################################################
#################################                 ANIMATION               #############################
#######################################################################################################
####################################################################################################### 

#PLOTTING PARAMETERS  
x_lim_list = []
y_lim_list = []
t_array = []

for i, eps in enumerate(elongation_list): 
    x_lim_list.append(separatrix_domain(a_list[i], b_list[i]))
    y_lim_list.append(separatrix_range(a_list[i], b_list[i]))
    t_array.append(np.linspace(-x_lim_list[i], x_lim_list[i], num = acc))
    
#GRAPHING PARAMETERS 
figAnimation, axis = plt.subplots()
axis.set_xlim(-(max(x_lim_list)), max(x_lim_list))
axis.set_ylim(-(max(y_lim_list)), max(y_lim_list))

positive_r,  = axis.plot([],[], color = "red")
negative_r, = axis.plot([],[], color = "red")

legend_text = axis.text(
    0.05, 0.95, "", transform=axis.transAxes, 
    fontsize=10, verticalalignment="top"
)


#GRPAHING FUNCTIONS 
def update_data(frame):    
    r_positive = separatrix_shape_pos(t_array[frame], a_list[frame], b_list[frame])
    r_negative = -r_positive
    positive_r.set_data(t_array[frame], r_positive)
    negative_r.set_data(t_array[frame], r_negative)
    
    shape_index = shape_index_N3(a_list[frame], b_list[frame], a_list[frame]/Rw)
    legend_text.set_text(f"Shape Index N3: {shape_index:.3f}")
    return positive_r, negative_r, legend_text

def update_limit(frame): 
    axis.set_xlim(x_lim_list[frame])
    axis.set_ylim(y_lim_list[frame])

animation = FuncAnimation(
                fig = figAnimation,
                func = update_data,
                frames = len(elongation_list),
                interval = 5
                          )

plt.xlabel("z [m]", fontsize = labelFontSize)                 # Label for the x-axis
plt.ylabel("r [m]", fontsize = labelFontSize)     # Label for the y-axis
plt.title("Separatrix Shape Animation w/ Shape Index")
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.legend(fontsize = 'medium')
plt.grid()
animation.save("separatrix_shape.gif")
plt.show()
