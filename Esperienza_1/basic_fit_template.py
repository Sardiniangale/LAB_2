import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Funzione per il modello lineare (Legge di Ohm: I = V/R -> I = a*V + b, con a = 1/R)
def linear_model(x, a, b):
    return a * x + b

# --- INSERISCI I TUOI DATI QUI ---
# Sostituisci i valori di esempio con le tue misure
delta_v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Valori di tensione (V)
sigma_delta_v = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # Incertezze sulla tensione (V)
i = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Valori di corrente (mA)
sigma_i = np.array([0.01, 0.01, 0.01, 0.01, 0.01])  # Incertezze sulla corrente (mA)
# --- FINE SEZIONE DATI ---


# Valori iniziali per il fit
p0 = [1.0, 0.0]

# Esecuzione del fit
popt, pcov = curve_fit(linear_model, delta_v, i, p0=p0, sigma=sigma_i, absolute_sigma=True)

# Parametri ottimali e loro incertezze
a_ott, b_ott = popt
sigma_a, sigma_b = np.sqrt(np.diag(pcov))

# Calcolo del chi-quadro
res = i - linear_model(delta_v, a_ott, b_ott)
chi2 = np.sum((res / sigma_i)**2)
ndof = len(delta_v) - len(popt)  # Gradi di libertà

print("--- RISULTATI DEL FIT ---")
print(f"Parametro 'a' (pendenza): {a_ott:.4f} +/- {sigma_a:.4f}")
print(f"Parametro 'b' (intercetta): {b_ott:.4f} +/- {sigma_b:.4f}")
print(f"Chi-quadro ridotto: {chi2/ndof:.4f}")
print("-------------------------")

# Grafico dei dati e del best-fit
plt.figure(figsize=(10, 6))
plt.errorbar(delta_v, i, yerr=sigma_i, xerr=sigma_delta_v, fmt='o', label='Dati sperimentali')
x_fit = np.linspace(min(delta_v), max(delta_v), 100)
y_fit = linear_model(x_fit, a_ott, b_ott)
plt.plot(x_fit, y_fit, '-', label='Best-fit lineare')
plt.xlabel("Differenza di potenziale (V)")
plt.ylabel("Corrente (mA)")
plt.title("Fit lineare dei dati sperimentali")
plt.legend()
plt.grid(True)
plt.show()

# Grafico dei residui normalizzati
plt.figure(figsize=(10, 4))
res_norm = res / sigma_i
plt.errorbar(delta_v, res_norm, yerr=np.ones_like(res_norm), fmt='o', label='Residui normalizzati')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Differenza di potenziale (V)")
plt.ylabel("Residui normalizzati")
plt.title("Grafico dei residui")
plt.legend()
plt.grid(True)
plt.show()