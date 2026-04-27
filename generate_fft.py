import numpy as np
import matplotlib.pyplot as plt
import os


def plotta_spettro(percorso_file, titolo, e_periodico):
    print("analizzo " + percorso_file)
    
    #carica i dati dal file di testo
    dati = np.loadtxt(percorso_file)
    
    #controlla quante colonne ci sono
    #se ci sono 3 o piu colonne il tempo e la prima e la tensione e la terza
    if dati.shape[1] >= 3:
        tempo = dati[:, 0]
        tensione = dati[:, 2]
    #altrimenti il tempo e la prima e la tensione e la seconda
    else:
        tempo = dati[:, 0]
        tensione = dati[:, 1]
        
    #calcola il passo temporale
    dt = tempo[1] - tempo[0]
    
    #se il passo e maggiore di 1 significa che e in microsecondi (dati arduino)
    if dt > 1:
        dt = dt * 1e-6 
        
    numero_punti = len(tensione)
    
    #calcola la trasformata di fourier (fft)
    spettro = np.abs(np.fft.rfft(tensione))
    #calcola le frequenze corrispondenti
    frequenze = np.fft.rfftfreq(numero_punti, d=dt)
    
    #trova il picco principale per tagliare il grafico
    #ignora il primo punto che e la componente continua
    indice_picco = np.argmax(spettro[1:]) + 1
    frequenza_picco = frequenze[indice_picco]
        
    #crea la figura
    plt.figure(figsize=(10, 6))
    
    if e_periodico == True:
        #grafico logaritmico per segnali periodici
        plt.semilogy(frequenze, spettro)
        #taglia l asse x a 25 armoniche o almeno 500 hz
        frequenza_max = max(frequenza_picco * 25, 500)
    else:
        #grafico lineare per oscillatore smorzato
        plt.plot(frequenze, spettro)
        #taglia l asse x vicino al picco per vedere la larghezza
        frequenza_max = max(frequenza_picco * 5, 1000)
        
    #assicurati di non superare la frequenza massima possibile
    frequenza_max = min(frequenza_max, frequenze[-1])
    plt.xlim(0, frequenza_max)
        
    #imposta i titoli e le etichette
    plt.title(titolo)
    plt.xlabel("Frequenza [Hz]")
    plt.ylabel("Magnitudine (u. a.)")
    
    #aggiungi la griglia
    plt.grid(True, which="both", ls="--", alpha=0.7)
        
    #salva il file nella cartella secondo_sem
    nome_file = os.path.basename(percorso_file)
    nome_immagine = nome_file.replace('.txt', '_fft.png')
    percorso_immagine = os.path.join("secondo_sem", nome_immagine)
    
    plt.savefig(percorso_immagine, bbox_inches='tight')
    plt.close()

#lista di tutti i file da analizzare
#formato: [percorso, titolo_grafico, e_periodico]
#file_da_analizzare

#ciclo per analizzare ogni file nella lista
for f in file_da_analizzare:
    plotta_spettro(f[0], f[1], f[2])
