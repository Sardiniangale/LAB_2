import numpy as np
import matplotlib.pyplot as plt

r = 10000
c = 100e-9
tau = r * c
f_cut = 1 / (2 * np.pi * tau)
freq = 500 
w0 = 2 * np.pi * freq
n_harm = 1000

#def time
t = np.linspace(0, 3/freq, 10000)

#definizione della funzione per "filtro"
def get_fish(t, n, w, tau):
    sig = np.zeros_like(t)
    
    #looping for:
    for k in range(1, n, 2):
        wk = k * w
        
        #coeff for square
        ak = 1/k
        
        #filter logic
        gain = 1 / np.sqrt(1 + (wk*tau)**2)
        phi = -np.arctan(wk*tau)
        
        #summing
        sig += ak * gain * np.sin(wk*t + phi)
        
    return sig

#get_fish makes the wave
#parte di sotto 

#fa ill high, low and cutoff
freqs_sim = [f_cut / 10, f_cut, f_cut * 10]



labels = ["Low Frequensa(0.1 * fc)-Square", "Cutoff Frequensa(fc)-Pinna di squalo", "High Frequensa(10 * fc)-Triangular"]

#ploting


plt.figure(figsize=(10, 10))
for i, f_s in enumerate(freqs_sim):
    w_s = 2 * np.pi * f_s
    
    #time
    t_s = np.linspace(0, 3/f_s, 5000)
    y_s = get_fish(t_s, n_harm, w_s, tau)
    #subplot
    plt.subplot(3, 1, i+1)
    plt.plot(t_s, y_s, label="S")
    plt.title(f"{labels[i]}")
    plt.xlabel("time [s]")
    plt.ylabel("voltage [arb]")
    plt.grid(True, which="both", ls="--")
    plt.legend()

plt.tight_layout(pad=3.0)
plt.show()
