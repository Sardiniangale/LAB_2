import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.stats as st

Filename=("/home/studentelab2/dir_mercoledi/Esperienza_11/25.02.26.GvsF.txt") #reads from here

#loding
f_j, V_in, delta_in, V_out, delta_out = np.loadtxt(Filename,unpack=True)

print(f_j, V_in, delta_in, V_out, delta_out)


G = V_out/V_in #gain

delta_G = (V_out * delta_in + V_in * delta_out)/V_in**2


#function
def gain(f_j, A, f_t):
    return A/np.sqrt(1+(f_j/f_t)**2)


#fit

popt, pcov = curve_fit(gain, f_j, G, sigma=delta_G, absolute_sigma=False)
A, f_t = popt
sigma_A, sigma_f_t = np.sqrt(np.diag(pcov))

#grafico

plt.figure()
plt.loglog(f_j, G, marker = '.', linestyle = 'None')
# plt.loglog(f_j, gain(f_j, *popt))
plt.xlabel("F [Hz]")
plt.ylabel("Gain")
plt.title("Gain VS F [Hz]")
plt.legend()
plt.show()