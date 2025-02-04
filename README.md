# LABORATORIO-1
Análisis Estaditico De La Señal
# Análisis Estadístico de Señales Fisiológicas

Este proyecto realiza un análisis estadístico y de ruido sobre una señal fisiológica (ECG) utilizando Python. A continuación, se detalla cada sección del código con explicaciones detalladas y fórmulas utilizadas.

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
Se utiliza `wfdb.rdrecord` para cargar la señal desde un archivo externo. Esta señal es un electrocardiograma (ECG) que contiene información fisiológica de interés. En este caso, se seleccionan los primeros 900 puntos para simplificar el análisis. Este método es estándar para el manejo de señales biomédicas almacenadas en formatos especializados como WFDB.

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
Se generan dos histogramas:
1. **Con Seaborn (`sns.histplot`)**: Incluye un ajuste de densidad (`kde=True`) para visualizar la distribución probabilística subyacente.
2. **Con Matplotlib (`plt.hist`)**: Muestra la frecuencia de valores en intervalos discretos.

Estas visualizaciones ayudan a comprender cómo están distribuidos los valores de la señal.

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
Se grafica la señal ECG en función del tiempo:
- **Eje X:** Representa el tiempo en milisegundos (ms).
- **Eje Y:** Muestra el voltaje en milivoltios (mV).

La gráfica incluye leyendas y cuadrículas para facilitar su interpretación.

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
Cálculo manual de:
- **Media (μ):** Representa el valor promedio de la señal.
- **Desviación Estándar (σ):** Mide la dispersión de los valores respecto a la media.
- **Coeficiente de Variación (CV):** Relaciona la desviación estándar con la media.

$$
\mu = \frac{\sum x_i}{n}, \quad
\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{n-1}}, \quad
CV = \frac{\sigma}{\mu}
$$



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
Aquí se usan funciones predefinidas de NumPy:
- `np.mean`: Calcula la media.
- `np.std`: Calcula la desviación estándar con `ddof=1` para muestra.

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
Se calcula la probabilidad de ocurrencia de cada valor único en la señal, lo que permite crear una distribución discreta:
\[
P(v) = \frac{\text{Frecuencia Absoluta de } v}{\text{Total de Valores}}
\]

---

### 6. Ruido Añadido y Cálculo de SNR
#### 6.1. Ruido Gaussiano
```python
ruido = np.random.normal(0, 0.1, t)
señal_ruidosa = señal + ruido
```
Se añade ruido con distribución normal (\( \mathcal{N}(0, 0.1) \)).

#### 6.2. Ruido de Impulso
```python
impulsos = np.random.choice([0, 1], size=t, p=[0.9, 0.1])
ruido_impulso = impulsos * np.random.uniform(-0.8, 0.8, t)
```
Se introducen impulsos aleatorios para simular eventos bruscos.

#### 6.3. Ruido Tipo Artefacto
```python
artefactos = np.random.choice([0, 1], size=t, p=[0.95, 0.05])
ruido_artefacto = artefactos * np.random.normal(5 * np.std(señal), 0.5, t)
```
Se simulan artefactos con alta desviación estándar (\( 5\sigma \)).

#### Cálculo del SNR
```python
def snr_gaussiano(s, r):
    snr = 10 * np.log10(np.mean(s**2) / np.mean(r**2))
    return snr

print('SNR con ruido gaussiano:', round(snr_gaussiano(señal, ruido), 3), 'dB')
```
Relación señal-ruido:
\[
SNR = 10 \cdot \log_{10} \left( \frac{P_{señal}}{P_{ruido}} \right)
\]

---

### 7. Visualización de Ruido
Se grafican las señales contaminadas con ruido gaussiano, de impulso y tipo artefacto, junto con la señal original:
```python
plt.figure()
plt.plot(señal, label='Señal original')
plt.plot(señal_ruidosa, label='Ruido gaussiano')
plt.legend()
plt.show()
```
Estas visualizaciones permiten observar cómo afecta cada tipo de ruido a la señal.

---

## Resultado Esperado
- Histogramas de la señal.
- Estadísticos descriptivos calculados manualmente y con NumPy.
- Función de probabilidad.
- Gráficas de señales contaminadas con ruido y sus SNR.

## Instrucciones
1. Descargar la señal desde bases de datos como PhysioNet.
2. Ejecutar el código en un entorno Python.
3. Subir este análisis a GitHub con el archivo `README.md` y las gráficas generadas como soporte visual.
