import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

Filename=("/home/studentelab2/dir_mercoledi/Esperienza 2/exp2.txt") #reads from here

#loding
R_j,dR_j,mI_j,dmI_j=np.loadtxt(Filename,unpack=True)

print(R_j,dR_j,mI_j,dmI_j)

#convesion
I_j = mI_j * 1e-6
dI_j = dmI_j * 1e-6

#model
def I_i(R_j, V_th, R_th):
    return V_th/(R_th + R_j)

#fiting

#prelim fit
popt_prelim, _ = curve_fit(I_i, R_j, I_j, p0=(1.0, 1000.0))
V_th_prelim, R_th_prelim = popt_prelim

#erorss
dI_dR = -V_th_prelim / (R_th_prelim + R_j)**2
effective_sigma = np.sqrt(dI_j**2 + (dI_dR * dR_j)**2)

#final fit
popt, pcov = curve_fit(I_i, R_j, I_j, sigma=effective_sigma, absolute_sigma=True)

V_th, R_th = popt
sV_th, sR_th = np.sqrt(np.diag(pcov))

#ploting
R_plot = np.logspace(np.log10(R_j.min()), np.log10(R_j.max()), 500)
I_plot = I_i(R_plot, V_th, R_th)

plt.figure(figsize=(10, 6))
plt.errorbar(R_j, I_j, xerr=dR_j, yerr=dI_j, fmt='o', label="Valori misurati", capsize=3, markersize=4)
plt.plot(R_plot, I_plot, label="Theoretical fit")

#log scal
plt.xscale("log")
plt.yscale("log")

#labels
plt.xlabel("Resistenze [Ohm]")
plt.ylabel("Corrente [A]")
plt.title("I vs R")
plt.grid(True, which="both", ls="--")
plt.legend()

#show
plt.show()

#ouput
print(f"ddp Thevenin: {V_th:.4f} +- {sV_th:.2f}")
print(f"Resistenza Thevenin:{R_th:.4f} +- {sR_th:.2f}")
