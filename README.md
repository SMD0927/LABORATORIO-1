# LABORATORIO-1
Análisis Estadístico de Señales Fisiológicas
# Análisis Estadístico de Señales Fisiológicas

Este proyecto realiza un análisis detallado de una señal fisiológica (ECG) utilizando técnicas estadísticas descriptivas y modelos de ruido, con el objetivo de evaluar características esenciales de la señal y los efectos del ruido en su comportamiento.

## Requisitos
- **Python 3.9**
- Bibliotecas necesarias:
  - `wfdb`
  - `numpy`
  - `matplotlib`
  - `seaborn`

Instalar dependencias:
```bash
pip install wfdb numpy matplotlib seaborn
```

## Estructura del Código

### 1. Lectura de Datos
```python
import wfdb
import numpy as np

datos = wfdb.rdrecord('rec_2')
t = 900
señal = datos.p_signal[:t, 0]
```
Se utiliza `wfdb.rdrecord` para cargar una señal fisiológica (ECG) de 900 puntos, obtenida de un archivo estándar en formato WFDB. Este tipo de señal es común en estudios biomédicos, donde se analiza la actividad eléctrica del corazón.

---

### 2. Histograma de la Señal
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(señal, kde=True, bins=30, color='red')
plt.hist(señal, bins=30, edgecolor='blue')
plt.title('Histograma de Datos')
plt.xlabel('Datos')
plt.ylabel('Frecuencia')
plt.show()
```
Se representa la distribución de la señal mediante histogramas. La gráfica muestra:
- **Frecuencia de los valores**: Observando la concentración de valores.
- **Densidad de probabilidad (KDE)**: Una curva suavizada que resalta patrones en la distribución.

---

### 3. Graficado de la Señal
```python
plt.figure(figsize=(10, 5))
plt.plot(señal, label="Señal fisiológica")
plt.title("ECG")
plt.xlabel("TIEMPO [ms]")
plt.ylabel("VOLTAJE [mV]")
plt.legend()
plt.grid()
plt.show()
```
Se grafica la señal original con etiquetas y leyendas. Esto permite identificar patrones, amplitudes y posibles irregularidades en el tiempo.

---

### 4. Estadísticos Descriptivos

#### 4.1. Cálculo Manual
```python
def estadisticos_programados():
    suma = 0
    for v in señal:
        suma += v    
    media = suma / t
    suma2 = sum((u - media)**2 for u in señal)
    desvesta = (suma2 / (t - 1))**0.5
    coeficiente = desvesta / media
    print('media:', media)
    print("desviacion estandar:", desvesta)
    print('coeficente de variacion', coeficiente)

estadisticos_programados()
```
Se calculan manualmente:
- **Media (μ):** Representa el promedio de los valores de la señal.
- **Desviación Estándar (σ):** Mide la dispersión de los datos respecto a la media.
- **Coeficiente de Variación (CV):** Relación entre la desviación estándar y la media, útil para datos comparativos.

Resultados típicos:
- Media: -0.0124
- Desviación estándar: 0.131
- Coeficiente de variación: -10.557

#### 4.2. Usando Funciones de NumPy
```python
def estadisticos_Bibliotecas():
    media = np.mean(señal)
    desvesta = np.std(señal, ddof=1)
    coeficiente = desvesta / media
    print('Media:', media)
    print("Desviación estándar:", desvesta)
    print('Coeficiente de variación:', coeficiente)

estadisticos_Bibliotecas()
```
NumPy permite calcular de manera eficiente los mismos estadísticos:
- Media: -0.012
- Desviación estándar: 0.131
- Coeficiente de variación: -10.554

---

### 5. Función de Probabilidad
```python
def calcular_funcion_probabilidad(senal):
    valores_unicos = np.unique(señal)
    probabilidades = {}
    for valor in valores_unicos:
        probabilidades[valor] = np.sum(señal == valor) / len(señal)
    for valor, prob in probabilidades.items():
        print(f"Valor: {valor:.5f}, Probabilidad: {prob:.5f}")

calcular_funcion_probabilidad(señal)
```
Se evalúa la frecuencia relativa de cada valor único en la señal:
- Ejemplo:
  - Valor: -0.28000, Probabilidad: 0.00050
  - Valor: 0.00000, Probabilidad: 0.01650

Esto facilita el análisis de patrones específicos en los datos.

---

### 6. Ruido Añadido y Cálculo de SNR
#### 6.1. Ruido Gaussiano
```python
ruido = np.random.normal(0, 0.1, t)
señal_ruidosa = señal + ruido
```
Se añade ruido normal (\( \mathcal{N}(0, 0.1) \)) para simular perturbaciones aleatorias.

#### 6.2. Ruido de Impulso
```python
impulsos = np.random.choice([0, 1], size=t, p=[0.9, 0.1])
ruido_impulso = impulsos * np.random.uniform(-0.8, 0.8, t)
```
Se introducen impulsos aleatorios, representando eventos transitorios bruscos.

#### 6.3. Ruido Tipo Artefacto
```python
artefactos = np.random.choice([0, 1], size=t, p=[0.95, 0.05])
ruido_artefacto = artefactos * np.random.normal(5 * np.std(señal), 0.5, t)
```
Artefactos que simulan eventos de alta amplitud (\( 5\sigma \)).

#### Cálculo del SNR
```python
def snr_gaussiano(s, r):
    snr = 10 * np.log10(np.mean(s**2) / np.mean(r**2))
    return snr

print('SNR con ruido gaussiano:', round(snr_gaussiano(señal, ruido), 3), 'dB')
```
Se evalúa la relación señal-ruido (SNR):
- **Ruido Gaussiano:** 10.369 dB
- **Ruido Impulso:** 7.09 dB
- **Ruido Artefacto:** 4.092 dB

---

### 7. Visualización de Ruido
Se grafican las señales contaminadas con ruido, permitiendo analizar el impacto visual de cada tipo de perturbación:
```python
plt.figure()
plt.plot(señal, label='Señal original')
plt.plot(señal_ruidosa, label='Ruido gaussiano')
plt.legend()
plt.show()
```

---

## Resultado Esperado
- Histogramas y estadísticos descriptivos.
- Distribución de probabilidad de valores.
- Gráficas de señales con diferentes tipos de ruido.
- Cálculo de SNR para evaluar la calidad de la señal bajo ruido.

## Instrucciones
1. Descargar la señal desde bases de datos como PhysioNet.
2. Ejecutar el código en un entorno Python.
3. Subir este análisis a GitHub con el archivo `README.md` y las gráficas generadas como soporte visual.

