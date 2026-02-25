import numpy as np
import matplotlib.pyplot as plt

r = 10000
c = 100e-9
tau = r * c
f_cut = 1 / (2 * np.pi * tau)
n_harm = 1000

def get_fish(t, n, w, tau):
    sig = np.zeros_like(t)
    for k in range(1, n, 2):
        wk = k * w
        ak = 1/k
        gain = 1 / np.sqrt(1 + (wk*tau)**2)
        phi = -np.arctan(wk*tau)
        sig += ak * gain * np.sin(wk*t + phi)
    return sig

# frequencies and files
# 30Hz, 480Hz(approx), 5000Hz(approx)
freqs_sim = [30.7, 482, 4820] 
files = ["merc/06.00.txt", "merc/06.04Q.txt", "merc/06.08Q-.txt"]
labels = ["Low Freq (30Hz) - Square", "Cutoff Freq (482Hz) - Shark Fin", "High Freq (4.8kHz) - Triangular"]

plt.figure(figsize=(10, 10))

for i, f_s in enumerate(freqs_sim):
    w_s = 2 * np.pi * f_s
    period = 1/f_s
    
    plt.subplot(3, 1, i+1)
    
    # default ranges
    t_plot = np.linspace(0, 3*period, 1000)
    
    # try load exp
    try:
        data = np.loadtxt(files[i])
        t_exp_raw = data[:, 0]
        v_exp_raw = data[:, 1]
        
        # 1. Handle Units (ms to s)
        t_exp = t_exp_raw
        if len(t_exp) > 1 and (t_exp[1] - t_exp[0]) > 0.001: 
             t_exp = t_exp / 1000.0

        # 2. Slice Data (Strictly 3 periods)
        t_exp = t_exp - t_exp[0] # Start at 0
        mask = t_exp < (3 * period)
        
        if np.sum(mask) > 10:
            t_exp = t_exp[mask]
            v_exp = v_exp_raw[mask]
            
            # 3. Robust Normalization (Fix "Flat" data)
            # Center around median to remove DC offset
            v_exp = v_exp - np.median(v_exp)
            # Scale by Robust Amplitude (95th percentile - 5th percentile)
            # This ignores outliers/spikes that squish the graph
            amp = np.percentile(v_exp, 95) - np.percentile(v_exp, 5)
            if amp == 0: amp = np.max(np.abs(v_exp)) # Fallback
            v_exp = v_exp / (amp / 2) # Approx normalize to [-1, 1]

            # 4. Downsample (Fix "Rectangle" block)
            if len(t_exp) > 1000:
                indices = np.linspace(0, len(t_exp)-1, 1000, dtype=int)
                t_exp = t_exp[indices]
                v_exp = v_exp[indices]

            # Plot Exp (BLUE DOTS)
            plt.plot(t_exp, v_exp, 'b.', markersize=3, alpha=0.5, label="Arduino Data")
            
            # Update sim range to match sliced exp
            t_plot = np.linspace(0, np.max(t_exp), 1000)

    except Exception as e:
        print(f"Skipping file {files[i]}: {e}")

    # Generate Simulation
    y_sim = get_fish(t_plot, n_harm, w_s, tau)
    # Normalize Sim
    y_sim = y_sim - np.median(y_sim)
    amp_sim = np.max(y_sim) - np.min(y_sim)
    y_sim = y_sim / (amp_sim / 2)
    
    # Plot Sim (Red Line)
    plt.plot(t_plot, y_sim, 'r-', linewidth=2, label="Fourier Sim")
    
    plt.title(labels[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Norm. Amplitude")
    plt.grid(True, ls="--")
    plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("shark_fin_comparison.png")
