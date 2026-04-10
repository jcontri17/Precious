#FRC 2D Equilibrium         Jeffrey D. Contri; jcontri@uw.edu           2025_08_20 - 
#Ref:   Cerfon and Freidberg (2010) doi: 10.1063/1.3328818



#################################################
#####               IMPORTS                 #####
#################################################
from __future__ import annotations
from functions4plasma import *
from plottingParameters import *
import numpy as np
import matplotlib.pyplot as plt
import cmath
import sys
import time




t0code03 = time.perf_counter()
def _debug():
    print("plottingParameters executed directly")

if __name__ == "__main__":
    _debug()
    
    #################################################
    #####               FUNCTIONS               #####
    #################################################

    #####------------------------------------------------------------FULL SOLUTION & DERIVATIVES

    def varphi_sol(x, y, A, c_array):
        varphipart = varphi_p(x, A)
        varphihomog = 0.0
        for i in range(len(c_array)):
            varphihomog += c_array[i] * varphi_funcs[i+1](x, y)
        varphi = varphipart + varphihomog
        return varphi


    def varphi_solx(x, y, A, c_array):
        varphipartx = varphi_px(x, A)
        varphihomogx = 0.0
        for i in range(len(c_array)):
            varphihomogx += c_array[i] * varphi_x_funcs[i+1](x, y)
        varphix = varphipartx + varphihomogx
        return varphix


    def varphi_solxx(x, y, A, c_array):
        varphipartxx = varphi_pxx(x, A)
        varphihomogxx = 0.0
        for i in range(len(c_array)):
            varphihomogxx += c_array[i] * varphi_xx_funcs[i+1](x, y)
        varphixx = varphipartxx + varphihomogxx
        return varphixx


    def varphi_soly(x, y, A, c_array):
        varphiparty = varphi_py(x, A)   # define if you have ∂φ_part/∂y
        varphihomogy = 0.0
        for i in range(len(c_array)):
            varphihomogy += c_array[i] * varphi_y_funcs[i+1](x, y)
            print(i, "varphi_y_func_i+1 = ", varphi_y_funcs[i+1](x,y), "c_i", c_array[i])
        print(varphiparty)
        varphiy = varphiparty + varphihomogy
        return varphiy


    def varphi_solyy(x, y, A, c_array):
        varphipartyy = varphi_pyy(x, A)   # define if you have ∂²φ_part/∂y²
        varphihomogyy = 0.0
        for i in range(len(c_array)):
            varphihomogyy += c_array[i] * varphi_yy_funcs[i+1](x, y)
        varphiyy = varphipartyy + varphihomogyy
        return varphiyy





    #####------------------------------------------------------PARTICULAR SOLUTION & DERIVATIVES
    xeps = 1e-12        #helps prevent the inf of ln(0)

    def varphi_p(x, A):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphipart = (x**4)/8.0 + A * (0.5 * x**2 * lnx - (x**4)/8.0)
        return varphipart


    def varphi_px(x, A):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphipartx = x**3/2.0 + A * (x*lnx + x/2.0 - x**3/2.0)
        return varphipartx


    def varphi_pxx(x, A):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphipartxx = (3/2)*x**2 + A * (lnx + 3/2 - (3/2)*x**2)
        return varphipartxx


    def varphi_py(x, A):
        return 0.0


    def varphi_pyy(x, A):
        return 0.0





    #####-------------------------------------------------------------------HOMOGENEOUS SOLUTION
    def varphi_1(x, y):
        varphi1 = 1
        return varphi1


    def varphi_2(x, y):
        varphi2 = x**2
        return varphi2


    def varphi_3(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi3 = y**2 - x**2 * lnx
        return varphi3


    def varphi_4(x, y):
        varphi4 = x**4 - 4.0 * x**2 * y**2
        return varphi4


    def varphi_5(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi5 = 2.0*y**4 - 9.0*y**2*x**2 + 3.0*x**4*lnx - 12.0*x**2*y**2*lnx
        return varphi5


    def varphi_6(x, y):
        varphi6 = x**6 - 12.0*x**4*y**2 + 8.0*x**2*y**4
        return varphi6


    def varphi_7(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi7 = (8.0*y**6
                - 140.0*y**4*x**2
                + 75.0*y**2*x**4
                - 15.0*x**6*lnx
                + 180.0*x**4*y**2*lnx
                - 120.0*x**2*y**4*lnx)
        return varphi7


    def varphi_8(x, y):
        varphi8 = y
        return varphi8


    def varphi_9(x, y):
        varphi9 = y * x**2
        return varphi9


    def varphi_10(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi10 = y**3 - 3.0*y*x**2*lnx
        return varphi10


    def varphi_11(x, y):
        varphi11 = 3.0*y*x**4 - 4.0*y**3*x**2
        return varphi11


    def varphi_12(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi12 = 8.0*y**5 - 45.0*y*x**4 - 80.0*y**3*x**2*lnx + 60.0*y*x**4*lnx
        return varphi12



    #####----------------------------------------------BOUNDARY CONDITION CURVATURE COEFFICIENTS
    def N1fun(eps, E, alpha):
        N1 = -((1.0 + alpha)**2) / (eps * (E**2))
        return N1


    def N2fun(eps, E, alpha):
        N2 = ((1.0 - alpha)**2) / (eps * (E**2))
        return N2


    def N3fun(eps, E, alpha):
        ca = np.cos(alpha)
        if np.isclose(ca, 0.0):
            raise ZeroDivisionError("cos(alpha) = 0 makes N3 singular.")
        N3 = - E / (eps * (ca**2))
        return N3



    #####-------------------------------------------------------HOMOGENEOUS SOLUTION DERIVATIVES

    ###---------------------------------------------------- ∂/∂x
    def varphi_1x(x, y):
        varphi1x = 0.0
        return varphi1x


    def varphi_2x(x, y):
        varphi2x = 2.0*x
        return varphi2x


    def varphi_3x(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi3x = -2.0*x*lnx - x
        return varphi3x


    def varphi_4x(x, y):
        varphi4x = 4.0*x**3 - 8.0*x*y**2
        return varphi4x


    def varphi_5x(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi5x = -30.0*x*y**2 + 12.0*x**3*lnx + 3.0*x**3 - 24.0*x*y**2*lnx
        return varphi5x


    def varphi_6x(x, y):
        varphi6x = 6.0*x**5 - 48.0*x**3*y**2 + 16.0*x*y**4
        return varphi6x


    def varphi_7x(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi7x = (-400.0*x*y**4 + 480.0*x**3*y**2 - 90.0*x**5*lnx
                    - 15.0*x**5 + 720.0*x**3*y**2*lnx - 240.0*x*y**4*lnx)
        return varphi7x


    def varphi_8x(x, y):
        varphi8x = 0.0
        return varphi8x


    def varphi_9x(x, y):
        varphi9x = 2.0*x*y
        return varphi9x


    def varphi_10x(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi10x = -6.0*x*y*lnx - 3.0*y*x
        return varphi10x


    def varphi_11x(x, y):
        varphi11x = 12.0*x**3*y - 8.0*x*y**3
        return varphi11x


    def varphi_12x(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi12x = -120.0*x**3*y - 160.0*x*y**3*lnx - 80.0*x*y**3 + 240.0*x**3*y*lnx
        return varphi12x



    ###-------------------------------------------------- ∂²/∂x²
    def varphi_1xx(x, y):
        varphi1xx = 0.0
        return varphi1xx


    def varphi_2xx(x, y):
        varphi2xx = 2.0
        return varphi2xx


    def varphi_3xx(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi3xx = -2.0*lnx - 3.0
        return varphi3xx


    def varphi_4xx(x, y):
        varphi4xx = 12.0*x**2 - 8.0*y**2
        return varphi4xx


    def varphi_5xx(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi5xx = -54.0*y**2 + 36.0*x**2*lnx + 21.0*x**2 - 24.0*y**2*lnx
        return varphi5xx


    def varphi_6xx(x, y):
        varphi6xx = 30.0*x**4 - 144.0*x**2*y**2 + 16.0*y**4
        return varphi6xx


    def varphi_7xx(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi7xx = (-640.0*y**4 
                    + 2160.0*x**2*y**2 
                    - 450.0*x**4*lnx
                    - 165.0*x**4 
                    + 2160.0*x**2*y**2*lnx 
                    - 240.0*y**4*lnx)
        return varphi7xx


    def varphi_8xx(x, y):
        varphi8xx = 0.0
        return varphi8xx


    def varphi_9xx(x, y):
        varphi9xx = 2.0*y
        return varphi9xx


    def varphi_10xx(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi10xx = -9.0*y - 6.0*y*lnx
        return varphi10xx


    def varphi_11xx(x, y):
        varphi11xx = 36.0*x**2*y - 8.0*y**3
        return varphi11xx


    def varphi_12xx(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi12xx = -120.0*x**2*y - 160.0*y**3*lnx - 240.0*y**3 + 720.0*x**2*y*lnx
        return varphi12xx



    ###---------------------------------------------------- ∂/∂y
    def varphi_1y(x, y):
        varphi1y = 0.0
        return varphi1y

    def varphi_2y(x, y):
        varphi2y = 0.0
        return varphi2y

    def varphi_3y(x, y):
        varphi3y = 2.0*y
        return varphi3y

    def varphi_4y(x, y):
        varphi4y = -8.0*x**2*y
        return varphi4y

    def varphi_5y(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi5y = 8.0*y**3 - 18.0*x**2*y - 24.0*x**2*y*lnx
        return varphi5y

    def varphi_6y(x, y):
        varphi6y = -24.0*x**4*y + 32.0*x**2*y**3
        return varphi6y

    def varphi_7y(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi7y = (48.0*y**5 
                    - 560.0*x**2*y**3 
                    + 150.0*x**4*y
                    + 360.0*x**4*y*lnx 
                    - 480.0*x**2*y**3*lnx)
        return varphi7y

    def varphi_8y(x, y):
        varphi8y = 1.0
        return varphi8y

    def varphi_9y(x, y):
        varphi9y = x**2
        return varphi9y

    def varphi_10y(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi10y = 3.0*y**2 - 3.0*x**2*lnx
        return varphi10y

    def varphi_11y(x, y):
        varphi11y = 3.0*x**4 - 12.0*x**2*y**2
        return varphi11y

    def varphi_12y(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi12y = 40.0*y**4 - 45.0*x**4 - 240.0*x**2*y**2*lnx + 60.0*x**4*lnx
        return varphi12y



    ###-------------------------------------------------- ∂²/∂y²
    def varphi_1yy(x, y):
        varphi1yy = 0.0
        return varphi1yy

    def varphi_2yy(x, y):
        varphi2yy = 0.0
        return varphi2yy

    def varphi_3yy(x, y):
        varphi3yy = 2.0
        return varphi3yy

    def varphi_4yy(x, y):
        varphi4yy = -8.0*x**2
        return varphi4yy

    def varphi_5yy(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi5yy = 24.0*y**2 - 18.0*x**2 - 24.0*x**2*lnx
        return varphi5yy

    def varphi_6yy(x, y):
        varphi6yy = -24.0*x**4 + 96.0*x**2*y**2
        return varphi6yy

    def varphi_7yy(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi7yy = (240.0*y**4 
                    - 1680.0*x**2*y**2 
                    + 150.0*x**4
                    + 360.0*x**4*lnx 
                    - 1440.0*x**2*y**2*lnx)
        return varphi7yy

    def varphi_8yy(x, y):
        varphi8yy = 0.0
        return varphi8yy

    def varphi_9yy(x, y):
        varphi9yy = 0.0
        return varphi9yy

    def varphi_10yy(x, y):
        varphi10yy = 6.0*y
        return varphi10yy

    def varphi_11yy(x, y):
        varphi11yy = -24.0*x**2*y
        return varphi11yy

    def varphi_12yy(x, y):
        if x==0:
            x = xeps
        lnx = np.log(np.abs(x))
        varphi12yy = 160.0*y**3 - 480.0*x**2*y*lnx
        return varphi12yy





    #################################################
    #####               DICTIONARIES            #####
    #################################################
    varphi_funcs = {
        1: varphi_1,
        2: varphi_2,
        3: varphi_3,
        4: varphi_4,
        5: varphi_5,
        6: varphi_6,
        7: varphi_7,
        8: varphi_8,
        9: varphi_9,
        10: varphi_10,
        11: varphi_11,
        12: varphi_12
    }

    varphi_x_funcs = {
        1: varphi_1x,
        2: varphi_2x,
        3: varphi_3x,
        4: varphi_4x,
        5: varphi_5x,
        6: varphi_6x,
        7: varphi_7x,
        8: varphi_8x,
        9: varphi_9x,
        10: varphi_10x,
        11: varphi_11x,
        12: varphi_12x
    }

    varphi_xx_funcs = {
        1: varphi_1xx,
        2: varphi_2xx,
        3: varphi_3xx,
        4: varphi_4xx,
        5: varphi_5xx,
        6: varphi_6xx,
        7: varphi_7xx,
        8: varphi_8xx,
        9: varphi_9xx,
        10: varphi_10xx,
        11: varphi_11xx,
        12: varphi_12xx
    }

    varphi_y_funcs = {
        1: varphi_1y,
        2: varphi_2y,
        3: varphi_3y,
        4: varphi_4y,
        5: varphi_5y,
        6: varphi_6y,
        7: varphi_7y,
        8: varphi_8y,
        9: varphi_9y,
        10: varphi_10y,
        11: varphi_11y,
        12: varphi_12y
    }

    varphi_yy_funcs = {
        1: varphi_1yy,
        2: varphi_2yy,
        3: varphi_3yy,
        4: varphi_4yy,
        5: varphi_5yy,
        6: varphi_6yy,
        7: varphi_7yy,
        8: varphi_8yy,
        9: varphi_9yy,
        10: varphi_10yy,
        11: varphi_11yy,
        12: varphi_12yy
    }



t1code03 = time.perf_counter()
##########################################################################
#####                           OUTPUT                               #####
##########################################################################
print("\n\n")
print("FRC_03: Grad Shafranov Functions & Dictrionaries =================")
print("\t frc02 Compute Time = {:.3e} [s]".format(t1code03-t0code03))









