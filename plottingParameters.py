import numpy as np
import matplotlib.pyplot as plt
import re
import sys
from matplotlib.ticker import FuncFormatter

figXsize = 12
figYsize = 8
labelFontSize = 14
titleFontSize = 18
tickFontSize = 12
textFontSize = 12
legendFontSize = 14
lineWidth = 1
fontWeight = 'bold'
dpi_res = 500


#################################################
#####               FUNCTIONS               #####
#################################################
###----------------------------------------------------------Forming Clean Axes Limit
def axesLimit(arr, sign):
    if sign==1:
        ma = np.abs(arr.max())
    elif sign==-1:
        ma = np.abs(arr.min())
    else:
        print("invalid sign (1 or -1) ---axesLimit()")
    
    x = 10**np.floor(np.log10(ma))
    y = 5*x*sign
    if (ma < np.abs(y)):
        lim = y
    else:
        lim = 2*y
    return lim


###---------------------------------------------------------Bold Expression
def bs(expr):
    return r'$\boldsymbol{' + s + '}$'

###------------------------------Scientific Notation Label
def sci_label(x, pos=None, bold=False, plus=False):
    if x == 0:
        return r'$\mathbf{0}$' if bold else '0'
    exp = int(np.floor(np.log10(abs(x))))
    coef = x / 10**exp
    sign = '+' if (plus and exp >= 0) else ''
    if bold:
        return r'$\mathbf{' + f'{coef:.1f}' + r'\mathbf{e}' + f'{sign}{exp}' + r'}$'
    else:
        s = f'{coef:.1f}e{sign}{exp}'
        s = s.replace('e-0', 'e-').replace('e+0', 'e+')
        return s


###---------------------------------------------------------No Padding on Sci Notation for Arrays (?)
def sci_no_pad(x, pos):
    s = f"{x:.1e}"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return s

###--------------------------------------------------------No Padding on Scientific Notation for Tick Label? Scinopad might be the same thing
def cbar_label(t):
    if t == 0:
        return r'$\mathbf{0}$'
    exp = int(np.floor(np.log10(abs(t))))
    coef = t / 10**exp
    sign = '+' if exp >= 0 else ''
    return r'$\mathbf{' + f'{coef:.1f}' + r'\mathbf{e}' + f'{sign}{exp}' + r'}$'


def _debug():
    print("plottingParameters executed directly")
    sys.exit()


if __name__ == "__main__":
    _debug()

    sys.exit()


    ###---------------------------------------------------------------------------------------SIMPLE FIGURE
    #figTitle = 'Number Density Profile\n1D $\\alpha$ Power Law'
    #saveTitle = str(re.sub(r'[\\/*?:"<>|\n$]', '_', figTitle))
    X_array = 0
    Y_array = 0

    X_min, X_max, X_step = 0, nodes[-1], 2
    Y_min, Y_max, Y_step = 0, 30, 5
    X_label = "x label [$x_u$]"
    Y_label = "y label [$y_u$]"


    fig1, ax1 = plt.subplots(figsize=(figXsize, figYsize))
    ax1.plot(X_array, Y_array, marker='o', color='blue')
    ax1.set_xlabel(X_label, fontsize=labelFontSize, fontweight=fontWeight)
    ax1.set_ylabel(Y_label, fontsize=labelFontSize, fontweight=fontWeight)
    ax1.set_title(figTitle, fontsize=labelFontSize, fontweight=fontWeight)
    ax1.set_xlim(X_min, X_max)
    ax1.set_ylim(Y_min, Y_max)
    ax1.set_xticks(np.arange(X_min, X_max+X_step, X_step))
    ax1.set_yticks(np.arange(Y_min, Y_max+Y_step, Y_step))
    ax1.tick_params(axis="both", which="major", labelsize=tickFontSize)
    ax1.grid(True)
    ax1.legend(location="best")
    fig1.tight_layout()
    plt.savefig(saveTitle, dpi=dpi_res)
    plt.show()




    ###-----------------------------------------------------------------------------------------------FIGURE
    #fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #fig, axes = plt.subplots(1, 4, figsize=(16, 8), constrained_layout=True)

    # Top plot: Pd and Pb
    #ax1.plot(Te_array, Pd, label='$P_d$')
    #ax1.plot(Te_array, Pb, label='$P_b$')
    #ax1.set_ylabel('Power')
    #ax1.legend()
    #ax1.set_title('Power Distribution: $P_d$ & $P_b$')

    # Bottom plot: Pwe, Pwi, Pa, Prion
    #ax2.plot(Te_array, Pwe,   label='$P_{we}$')
    #ax2.plot(Te_array, Pwi,   label='$P_{wi}$')
    #ax2.plot(Te_array, Pa,    label='$P_a$')
    #ax2.plot(Te_array, Prion, label='$P_{rad}+P_{ion}$')
    #ax2.set_xlabel('$T_e$')
    #ax2.set_ylabel('Power')
    #ax2.legend()
    #ax2.set_title('Power Distribution: $P_{we}$, $P_{wi}$, $P_a$, $P_{rad}+P_{ion}$')

    #fig.tight_layout()
    #plt.show()


    sys.exit()
    #################################################
    #####               FEATURES                #####
    #################################################
    ###---------------------------------------------------------CLEAN FIGURE TITLE AND SAVED TITLE
    #figTitle = 'Number Density Profile\n1D $\\alpha$ Power Law'
    #saveTitle = str(re.sub(r'[\\/*?:"<>|\n$]', '_', figTitle))


    ###-------------------------------------------------------BOLD LEGEND FONT (LEGEND PROPERTIES)
    #legprop = {'weight': 'bold'}
    #axs[0].legend(prop=legprop)


    ###----------------------------------------------------------------BLACK BACKGROUND
    #fig.patch.set_facecolor("black")   # outside area
    #ax.set_facecolor("black")          # plot area
    #cbar.ax.set_facecolor("black")     # colorbar background


    ###----------------------------------------------------------WHITE TICKS/LABELS
    #ax.tick_params(colors="white")
    #cbar.ax.yaxis.set_tick_params(color="white")
    #plt.setp(cbar.ax.get_yticklabels(), color="white")


    ###--------------------------------------------------------------BOLD TICK LABELS
    ax.tick_params(axis='both', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')


    ###-------------------------------------------------------------------------RAINBOW GRADIENT LINES
    numColors = 5
    colors = plt.cm.rainbow(np.linspace(0, 1, numColors))




