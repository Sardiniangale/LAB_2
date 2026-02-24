import numpy as np
import matplotlib.pyplot as plt

# Circuito Derivatore (Differentiator)
# High-pass filter: CR circuit (Voltage across Resistor)
# Transfer function H(w) = jwRC / (1 + jwRC)
# Gain G(w) = wRC / sqrt(1 + (wRC)^2)
# Phase phi = pi/2 - arctan(wRC)

r = 10000
c = 100e-9
tau = r * c
f_cut = 1 / (2 * np.pi * tau)
n_harm = 1000

def get_differentiator_output(t, n, w, tau, wave_type="square"):
    sig = np.zeros_like(t)
    
    for k in range(1, n, 2):
        wk = k * w
        
        if wave_type == "square":
            ak = 1/k
        elif wave_type == "triangle":
             ak = (2 / (k * np.pi))**2 # Normalized factor, ignoring DC
             # Real triangle series is 8/pi^2 sum (-1)^((k-1)/2) / k^2 sin(kt) usually
             # Let's stick to square as requested mostly.
             pass

        # Differentiator Transfer Function
        # H(jw) = j(wk*tau) / (1 + j(wk*tau))
        # Mag = (wk*tau) / sqrt(1 + (wk*tau)^2)
        # Arg = pi/2 - arctan(wk*tau)
        
        mag = (wk * tau) / np.sqrt(1 + (wk * tau)**2)
        phi = np.pi/2 - np.arctan(wk * tau)
        
        sig += ak * mag * np.sin(wk * t + phi)
        
    return sig

# Frequencies to simulate
freqs_sim = [f_cut / 10, f_cut, f_cut * 10]
labels = ["f << fc (Derivative)", "f = fc", "f >> fc (Pass-through)"]

plt.figure(figsize=(10, 10))

for i, f_s in enumerate(freqs_sim):
    w_s = 2 * np.pi * f_s
    t_s = np.linspace(0, 3/f_s, 5000)
    
    # Compute output
    y_s = get_differentiator_output(t_s, n_harm, w_s, tau)
    
    # Normalize for plotting
    y_s = y_s / np.max(np.abs(y_s))
    
    plt.subplot(3, 1, i+1)
    plt.plot(t_s, y_s, label="Simulation")
    plt.title(f"Derivatore: {labels[i]} (f={f_s:.1f} Hz)")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [arb]")
    plt.grid(True, ls="--")
    plt.legend()

plt.tight_layout()
plt.savefig("differentiator_sim.png")
# plt.show()
