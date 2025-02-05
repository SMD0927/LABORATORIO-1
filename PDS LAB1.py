import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



datos = wfdb.rdrecord('rec_2') 
#anotaciones = wfdb.rdann('rec_2', 'atr')  
t = 2000
señal = datos.p_signal[:t, 0]  

sns.histplot(señal, kde=True, bins=30, color='red')
plt.hist(señal, bins=30, edgecolor='blue')
plt.title('Histograma de Datos')
plt.xlabel('datos')
plt.ylabel('Frecuencia')
plt.show()


fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal, label="Señal fisiológica")
plt.title("ECG")  
plt.xlabel("TIEMPO[ms]") 
plt.ylabel("VOLTAJE[mv]") 
plt.legend() 
plt.grid() 
plt.show()

def estadisticos_programados():
    print('estadísticos descriptivos, forma larga;')
    suma = 0
    for v in señal:
       suma += v    
    media = suma/t
    print('media:', round(media,4))
    suma2 = 0
    
    for u in señal:
        suma2 += ((u-media)**2)
    
    desvesta = (suma2/(t-1))**0.5
    print("desviacion estadar:", round(desvesta,3))
    coeficiente = desvesta/media
    print('coeficente de variacion', abs(round(coeficiente,3)))
    print()


def estadisticos():
    print('estadísticos descriptivos, funciones predeterminadas;')
    media = np.mean(señal)
    desvesta = np.std(señal)
    coeficiente = desvesta/media
    print('media:', round(media,3))
    print("desviacion estadar:", round(desvesta,3))
    print('coeficente de variacion',abs(round(coeficiente,3)))

print()

def calcular_funcion_probabilidad(senal):
    valores_unicos = np.unique(señal)
    probabilidades = {}
    for valor in valores_unicos:
        probabilidades[valor] = np.sum(señal == valor) / len(señal)
    for valor, prob in probabilidades.items():
        print(f"Valor: {valor:.5f}, Probabilidad: {prob:.5f}")

calcular_funcion_probabilidad(señal)
print()
def snr(s,r):
    potencia_señal = np.mean(s**2)
    potencia_ruido = np.mean(r**2)
    
    if potencia_ruido == 0:
        return np.inf
    snr = 10 * np.log10(potencia_señal/potencia_ruido) 
    return snr 


#ruido gaussiano
ruido = np.random.normal(0, 0.04, t) 
señal_ruidosa = señal + ruido 
fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal_ruidosa, label="Ruido gaussiano", color='red')
plt.plot(señal, label="Señal fisiológica", color='black')
plt.xlabel("TIEMPO [ms]") 
plt.ylabel("VOLTAJE [mv]") 
plt.title(f"SEÑAL CON RUIDO GAUSSIANO, Snr = {round(snr(señal,ruido),3)} dB", fontsize=14)
plt.legend() 
plt.grid() 


#ruido de impulso
prob_impulso = 0.08
impulsos = np.random.choice([0, 1], size=len(señal), p=[1-prob_impulso, prob_impulso])
amplitud_impulso = np.random.choice([-1, 1], size=len(señal)) * 0.2
ruido2 = impulsos * amplitud_impulso
señal_con_ruido = señal + ruido2
fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal_con_ruido, label="Ruido de impulso", color='green')
plt.plot(señal, label="Señal fisiológica", color='black')
plt.title(f"SEÑAL CON RUIDO IMPULSO, Snr = {round(snr(señal,ruido2),3)} dB", fontsize=14)
plt.xlabel("TIEMPO [ms]") 
plt.ylabel("VOLTAJE [mv]") 
plt.legend() 
plt.grid() 


# ruido artefacto
prob_imp = 0.15
impul = np.random.choice([0, 1], size=len(señal), p=[1-prob_imp, prob_imp])
amplitud = np.random.choice([-1, 1], size=len(señal)) * 0.2
ruido3 = impul * amplitud
señal_ruido = señal + ruido3
fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal_ruido, label="Ruido de artefacto", color='magenta')
plt.plot(señal, label="Señal fisiológica", color='black')
plt.title(f"SEÑAL CON RUIDO ARTEFACTO, Snr = {round(snr(señal,ruido3),3)} dB", fontsize=14)
plt.xlabel("TIEMPO [ms]") 
plt.ylabel("VOLTAJE [mv]") 
plt.legend() 
plt.grid() 


#ruido gaussiano amplificado
ruido4 = np.random.normal(0, 0.1, t) 
señal_ruidosa = señal + ruido4 
fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal_ruidosa, label="Ruido gaussiano amplificado", color='red')
plt.plot(señal, label="Señal fisiológica", color='black')
plt.xlabel("TIEMPO [ms]") 
plt.ylabel("VOLTAJE [mv]") 
plt.title(f"SEÑAL CON RUIDO GAUSSIANO AMPLIFICADO, Snr = {round(snr(señal,ruido4),3)} dB", fontsize=14)
plt.legend() 
plt.grid() 

#ruido de impulso amplificado
prob = 0.08
im = np.random.choice([0, 1], size=len(señal), p=[1-prob, prob])
am = np.random.choice([-1, 1], size=len(señal)) * 0.4
ruido5 = im * am
señal_con_ruido = señal + ruido5
fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal_con_ruido, label="Ruido de impulso amplificado", color='green')
plt.plot(señal, label="Señal fisiológica", color='black')
plt.title(f"SEÑAL CON RUIDO IMPULSO AMPLIFICADO, Snr = {round(snr(señal,ruido5),3)} dB", fontsize=14)
plt.xlabel("TIEMPO [ms]") 
plt.ylabel("VOLTAJE [mv]") 
plt.legend() 
plt.grid() 

# ruido artefacto amplificado
p = 0.2
i = np.random.choice([0, 1], size=len(señal), p=[1-p, p])
a = np.random.choice([-1, 1], size=len(señal)) * 0.4
ruido6 = i * a
señal_ruido = señal + ruido6
fig = plt.figure(figsize=(10, 5)) 
plt.plot(señal_ruido, label="Ruido de artefacto amplificado", color='magenta')
plt.plot(señal, label="Señal fisiológica", color='black')
plt.title(f"SEÑAL CON RUIDO ARTEFACTO AMPLIFICADO, Snr = {round(snr(señal,ruido6),3)} dB", fontsize=14)
plt.xlabel("TIEMPO [ms]") 
plt.ylabel("VOLTAJE [mv]") 
plt.legend() 
plt.grid() 
plt.show()

estadisticos_programados()
estadisticos()
print()
print('SNR con ruido gaussiano:',  round(snr(señal,ruido),3), 'dB')
print('SNR con ruido impulso:',  round(snr(señal,ruido2),3), 'dB')
print('SNR con ruido artefacto:',  round(snr(señal,ruido3),3), 'dB')
print('SNR con ruido gaussiano amplificado:',  round(snr(señal,ruido4),3), 'dB')
print('SNR con ruido impulso amplificado:',  round(snr(señal,ruido5),3), 'dB')
print('SNR con ruido artefacto amplificado:',  round(snr(señal,ruido6),3), 'dB')


