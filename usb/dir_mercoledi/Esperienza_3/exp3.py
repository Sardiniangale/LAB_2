import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy.stats as st

Filename=("/home/studentelab2/dir_mercoledi/Esperienza_3/exp3.txt") #reads from here

#loding
V_dig, V_v, sigma_dig, sigma_v=np.loadtxt(Filename,unpack=True)

print(V_dig, V_v, sigma_dig, sigma_v)


#model
def linear_model(V_dig, a, b):
    return a + b * V_dig

def quadratic_model(V_dig, c, d, e):
    return c + d * V_dig + e * V_dig**2


#prelim fit
popt_prelim, pcov_prelim= curve_fit(linear_model, V_dig, V_v, p0=None)
a_prelim, b_prelim = popt_prelim
print(b_prelim)
#errors
effective_sigma_L = np.sqrt(sigma_v**2 + (sigma_dig * b_prelim)**2)

#final fit
popt, pcov = curve_fit(linear_model, V_dig, V_v, sigma=effective_sigma_L, absolute_sigma=False)

a, b = popt
sigma_a, sigma_b = np.sqrt(np.diag(pcov))

#ploting
# plt.figure(figsize=(10, 6))
# plt.errorbar(V_dig, V_v, xerr=None, yerr=effective_sigma_L, fmt='o', label="Valori misurati", capsize=3, markersize=4)
# plt.plot(V_dig, linear_model(V_dig, a, b), label="Theoretical fit")
#
# #labels
# plt.xlabel("V digitalizzata [digit]")
# plt.ylabel("V misurata [V]")
# plt.title("V misurata vs digitallizata")
# plt.grid(True, which="both", ls="--")
# plt.legend()

#quadratic model

#prelim fit/home/studentelab2/dir_mercoledi/Esperienza_3/exp3.txt
popt_prelim, pcov_prelim= curve_fit(quadratic_model, V_dig, V_v, p0=None)
c_prelim, d_prelim, e_prelim = popt_prelim

#errors
effective_sigma = np.sqrt(sigma_v**2 + (sigma_dig * (d_prelim + e_prelim* V_dig)**2))

#final fit
popt_q, pcov_q = curve_fit(quadratic_model, V_dig, V_v, sigma=effective_sigma, absolute_sigma=False)
c, d, e = popt_q
sigma_c, sigma_d, sigma_e = np.sqrt(np.diag(pcov_q))

#ploting
plt.figure(figsize=(10, 6))
plt.errorbar(V_dig, V_v, xerr=None, yerr=effective_sigma, fmt='o', label="Valori misurati", capsize=3, markersize=4)
plt.plot(V_dig, quadratic_model(V_dig, c, d, e), color="blue", label="Theoretical quadratic fit")
plt.plot(V_dig, linear_model(V_dig, a, b), color="red", label="Theoretical linear fit")

#labels
plt.xlabel("V digitalizzata [digit]")
plt.ylabel("V misurata [V]")
plt.title("V misurata vs digitallizata")
plt.grid(True, which="both", ls="--")
plt.legend()


#show
plt.show()

#ouput
print(f"a: {a:.4f} +- {sigma_a:.6f}")
print(f"b:{b:.4f} +- {sigma_b:.6f}")
print(f"c: {c:.4f} +- {sigma_c:.6f}")
print(f"d: {d:.4f} +- {sigma_d:.6f}")
print(f"e: {e:.9f} +- {sigma_e:.9f}")


# Calcolo del chi^2 caso lineare
V_v_fit = linear_model(V_dig, a, b)
chi_2_l = np.sum(((V_v - V_v_fit) / sigma_v)**2)
ndof = len(V_v) - 2
chi_red_l = chi_2_l / ndof

# Calcolo del chi^2 caso quadratico
V_v_fit = quadratic_model(V_dig, c, d, e)
chi_2_q = np.sum(((V_v - V_v_fit) / sigma_v)**2)
ndof = len(V_v) - 3
chi_red_q = chi_2_q / ndof

# Print results
print(f"Chi_2 modello lineare = {chi_2_l:.3f}")
print(f"Chi_2 ridotto modello lineare = {chi_red_l:.3f}")
print(f"Chi_2 modello quadratico = {chi_2_q:.3f}")
print(f"Chi_2 ridotto modello quadratico = {chi_red_q:.3f}")

# Istogramma e fit della gaussiana
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Istogramma con bin di ampiezza unitaria
t, V_dig = np.loadtxt("/home/studentelab2/dir_mercoledi/Esperienza_3/data6mer.txt", unpack=True)
counts, bin_edges = np.histogram(V_dig, bins=np.arange(min(V_dig), max(V_dig) + 1, 1))
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Fit gaussiano
A0 = np.max(counts)
mu0 = np.mean(V_dig)
sigma0 = np.std(V_dig)
p0 = [A0, mu0, sigma0]

popt_gauss, pcov_gauss = curve_fit(gaussian, bin_centers, counts, p0=p0)
A, mu, sigma = popt_gauss
sigma_A, sigma_mu, sigma_sigma = np.sqrt(np.diag(pcov_gauss))

print("\n--- Fit Gaussiano sui V_dig ---")
print(f"Media (mu) = {mu:.4f} ± {sigma_mu:.4f}")
print(f"Deviazione standard (sigma) = {sigma:.4f} ± {sigma_sigma:.4f}")

# Grafico
plt.figure()
plt.bar(bin_centers, counts, width=1, alpha=0.6, label='Istogramma V_dig')
x_fit = np.linspace(min(V_dig), max(V_dig), 500)
plt.plot(x_fit, gaussian(x_fit, *popt_gauss), 'r-', label='Fit gaussiano')
plt.xlabel("V_dig")
plt.ylabel("Occorrenze")
plt.title("Distribuzione dei valori digitali")
plt.legend()
plt.show()
