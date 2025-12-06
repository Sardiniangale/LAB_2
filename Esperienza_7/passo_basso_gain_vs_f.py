import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.stats as st

Filename=("/home/studentelab2/dir_mercoledi/Esperienza_7/passo_basso_gain.txt") #reads from here

#loding
f_j, V_in_1, delta_in, V_out_1, delta_out, V_in_2, sigma_in, V_out_2, sigma_out=np.loadtxt(Filename,unpack=True)

print(f_j, V_in_1, delta_in, V_out_1, delta_out, V_in_2, sigma_in, V_out_2, sigma_out)


G_1 = V_out_1/V_in_1 #gain aquizisione 1 (non gaussiana, scala diversa)
G_2 = V_out_2/V_in_2 #gain aquizisione 2 (gaussiana, scala uguale)

delta_G1 = (V_out_1 * delta_in + V_in_1 * delta_out)/V_in_1**2
delta_G2 = (V_out_2 * sigma_in + V_in_2 * sigma_out)/V_in_2**2

#function
def gain(f_j, A, f_t):
    return A/np.sqrt(1+(f_j/f_t)**2)


#fit_1 (non gaussiana)

popt_1, pcov_1 = curve_fit(gain, f_j, G_1, sigma=delta_G1, absolute_sigma=False)
A, f_t = popt_1
sigma_A, sigma_f_t = np.sqrt(np.diag(pcov_1))



#fit_2 (gaussiana)

popt_2, pcov_2 = curve_fit(gain, f_j, G_2, sigma=delta_G2, absolute_sigma=True)
A, f_t = popt_2
sigma_A, sigma_f_t = np.sqrt(np.diag(pcov_2))

#grafico

plt.figure()
plt.loglog(f_j, G_1, marker = 'o', linestyle = '-')
plt.loglog(f_j, gain(f_j, *popt_1))
plt.xlabel("F [Hz]")
plt.ylabel("Gain")
plt.title("Gain VS F [Hz]")
plt.legend()
plt.show()