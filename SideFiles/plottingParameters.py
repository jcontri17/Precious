import numpy as np
import matplotlib.pyplot as plt
import re


figXsize = 12
figYsize = 8
labelFontSize = 14
titleFontSize = 18
textFontSize = 12
legendFontSize = 14
lineWidth = 1
dpi_res = 500


colors = plt.cm.rainbow(np.linspace(0, 1, 5))
#figTitle = 'Number Density Profile\n1D $\\alpha$ Power Law'
#saveTitle = str(re.sub(r'[\\/*?:"<>|\n$]', '_', figTitle))





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




