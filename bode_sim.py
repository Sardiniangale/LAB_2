import numpy as np
import matplotlib.pyplot as plt

#config
r = 10000
c = 100e-9
tau = r * c
f_cut = 1 / (2 * np.pi * tau)
n_harm = 1000

#frequency
#da 10 a 1000khz
freqs = np.logspace(1, 5, 50)

#theo sin model
def sine_gain_theory(f, tau):
    w = 2 * np.pi * f
    return 1 / np.sqrt(1 + (w * tau)**2)

#simulazione for sqare
def get_square_gain_sim(freqs, n_harm, tau):
    gains = []
    
    for f in freqs:
        w = 2 * np.pi * f
        #create one period of time
        t = np.linspace(0, 1/f, 1000)
        
        #construct output signal (shark fin / triangle)
        sig = np.zeros_like(t)
        
        #sum harmonics
        for k in range(1, n_harm, 2):
            wk = k * w
            ak = 1/k #square wave coeff
            
            #filter
            g = 1 / np.sqrt(1 + (wk*tau)**2)
            phi = -np.arctan(wk*tau)
            
            sig += ak * g * np.sin(wk*t + phi) 

        #Idealmente square wave va da -pi/4 to +pi/4 * 4/pi 
        #4/pi * sum(sin(kwt)/k). Quindi peak amplitude is 1
        #sum(1/k * sin)
        #peak of sum(1/k sin(kx)) for square wave is pi/4
        #qundi input amplitude is pi/4
        
        input_amp = np.pi / 4
        
        #measure output amplitude
        out_amp = (np.max(sig) - np.min(sig)) / 2
        gains.append(out_amp / input_amp)
        
    return np.array(gains)

#compute
y_sine = sine_gain_theory(freqs, tau)
y_square = get_square_gain_sim(freqs, n_harm, tau)

#load experimental stuff
#Columns: f (Hz), Vout (V), Sigma_out, Vin (V), Sigma_in
try:
    data = np.loadtxt("Esperienza_6/gain.txt")
    print(f"Loaded {len(data)} experimental data points.")
    f_exp = data[:, 0]
    v_out = data[:, 1]
    v_in = data[:, 3]
    #calculate gain
    gain_exp = v_out / v_in
except Exception as e:
    print(f"Error loading gain.txt: {e}")
    f_exp, gain_exp = [], []

#ploting
plt.figure(figsize=(10, 6))

#theoretical sine
plt.plot(freqs, y_sine, label="Theoretical Sinusoidale", color="blue", linestyle="-")

#simulated square
plt.plot(freqs, y_square, label="Simulated Onda quadra", color="green", linestyle="--")

#experimental dots
if len(f_exp) > 0:
    plt.scatter(f_exp, gain_exp, color="red", marker='o', s=25, label="Experimental Data", zorder=5)

#cutoff marker
plt.axvline(x=f_cut, color="black", linestyle=":", label=f"f_cut = {f_cut:.1f} Hz")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequenza [Hz]")
plt.ylabel("Gain (V_out / V_in)")
plt.title("Integratore with Onda Quadra vs Sinusoidale in Bode plot")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend()
plt.savefig("bode_plot_corrected.png")
# plt.show()
