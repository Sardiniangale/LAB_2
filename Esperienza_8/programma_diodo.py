import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.stats as st

# Dati
Filename1=("/home/studentelab2/dir_mercoledi/Esperienza_8/data.0.3_CH1_V.txt")
Filename2=("/home/studentelab2/dir_mercoledi/Esperienza_8/data.0.3_CH2_V.txt")

t1, delta_t1, V1, delta_V1=np.loadtxt(Filename1,unpack=True)
t2, delta_t2, V2, delta_V2=np.loadtxt(Filename2,unpack=True)

print(t1, delta_t1, V1, delta_V1)
print(t2, delta_t2, V2, delta_V2)

Rd = 363
delta_R = 4
DeltaV = (V1-V2)
sigma_V=(delta_V1 + delta_V2)

# Funzione
def diodo(DeltaV, I_s, a):
    return I_s*(np.exp(DeltaV/a)-1)

I = (DeltaV)/Rd
sigma_I = (DeltaV*delta_R+Rd*(sigma_V))/Rd**2
print(I, sigma_I)
# Fit

popt, pcov = curve_fit(diodo, DeltaV, I, sigma=sigma_I, absolute_sigma=True)
I_s, a = popt
sigma_Is, sigma_a = np.sqrt(np.diag(pcov))
print(f"I_s={I_s:.4f} +- {sigma_Is:4f}")
print(f"a={a:.4f} +- {sigma_a:4f}")
# Grafico

plt.figure()
plt.errorbar(V2, I, yerr=sigma_I, marker = '.', label="Dati")
plt.plot(V2, diodo(DeltaV, *popt), "r--", label="Fit")
plt.xlabel("Delta V [V]")
plt.ylabel("I [A]")
plt.title("Corrente nel diodo VS Delta V")
plt.legend()
plt.grid(True)
plt.show()