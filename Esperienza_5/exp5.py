import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.stats as st

Filename=("/home/studentelab2/dati_arduino/merc/5.1.txt") #reads from here

#loding
t, sigma_t, V, sigma_V =np.loadtxt(Filename,unpack=True) #microseconds & digits

print(t, sigma_t, V, sigma_V)

#define

def carica_con(t, tau, a, b):
    return a - b *(np.exp(-t/tau))

def scarica_con(t, tau, a, b):
    return a*(np.exp(-t/tau))+b

#Carica condensatore 5.1

# Fit
popt, pcov = curve_fit(carica_con, t, V, sigma=sigma_V, absolute_sigma=True)

tau, a, b = popt
sigma_tau, sigma_a, sigma_b = np.sqrt(np.diag(pcov))

#ploting
fig=plt.figure(figsize=(10, 6))
ax1, ax2=fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(t, V, xerr=None, yerr=sigma_V, fmt='o', label="Valori misurati", capsize=3, markersize=4)
ax1.plot(t, carica_con(t, tau, a, b), color="blue", label="Theoretical fit")

#Residuals
residui = V-carica_con(t, *popt)
ax2.errorbar(t, residui, xerr=None, yerr=sigma_V, fmt='o', label="Residui su V", capsize=3, markersize=4)
ax2.plot(t, np.full(t.shape, 0))

#labels
ax1.set_title("V vs t")
ax1.set_xlabel("t [micro seconds]")
ax1.set_ylabel("V [digit]")
ax2.set_xlabel("t [micro seconds]")
ax2.set_ylabel("V [digit]")
ax1.grid(True, which="both", ls="--")
ax2.grid(True, which="both", ls="--")
plt.legend()


#show
plt.show()

#ouput
print("Carica")
print(f"a: {a:.4f} +- {sigma_a:.6f}")
print(f"b:{b:.4f} +- {sigma_b:.6f}")
print(f"tau: {tau:.4f} +- {sigma_tau:.6f}")


# Calcolo del chi^2 caso lineare
V_fit = carica_con(t, *popt)
chi_2 = np.sum(((V - V_fit))**2 / sigma_V)
ndof = len(V) - len(popt) - 1
chi_2_red = chi_2 / ndof


# Print results
print(f"Chi_2 carica = {chi_2:.3f}")
print(f"Chi_2 Ridoto carica = {chi_2_red:.3f}")

#Scarica condensatore 5.2
Filename2=("/home/studentelab2/dati_arduino/merc/5.2.txt") #reads from here
t, sigma_t, V, sigma_V =np.loadtxt(Filename2,unpack=True)

# Fit
popt, pcov = curve_fit(scarica_con, t, V, sigma=sigma_V, absolute_sigma=True)

tau, a, b = popt
sigma_tau, sigma_a, sigma_b = np.sqrt(np.diag(pcov))

#ploting
fig=plt.figure(figsize=(10, 6))
ax1, ax2=fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))
ax1.errorbar(t, V, xerr=None, yerr=sigma_V, fmt='o', label="Valori misurati", capsize=3, markersize=4)
ax1.plot(t, scarica_con(t, tau, a, b), color="blue", label="Theoretical fit")

#Residuals
residui = V-scarica_con(t, *popt)
ax2.errorbar(t, residui, xerr=None, yerr=sigma_V, fmt='o', label="Residui su V", capsize=3, markersize=4)
ax2.plot(t, np.full(t.shape, 0))

#labels
ax1.set_title("V vs t")
ax1.set_xlabel("t [micro seconds]")
ax1.set_ylabel("V [digit]")
ax2.set_xlabel("t [micro seconds]")
ax2.set_ylabel("V [digit]")
ax1.grid(True, which="both", ls="--")
ax2.grid(True, which="both", ls="--")
plt.legend()


#show
plt.show()

#ouput
print("Scarica")
print(f"a: {a:.4f} +- {sigma_a:.6f}")
print(f"b:{b:.4f} +- {sigma_b:.6f}")
print(f"tau: {tau:.4f} +- {sigma_tau:.6f}")



# Calcolo del chi^2 caso lineare
V_fit = scarica_con(t, *popt)
chi_2 = np.sum(((V - V_fit))**2 / sigma_V)
ndof = len(V) - len(popt) - 1
chi_2_red = chi_2 / ndof


# Print results
print(f"Chi_2 Scarica = {chi_2:.3f}")
print(f"Chi_2 Ridoto Scarica = {chi_2_red:.3f}")


