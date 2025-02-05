# LABORATORIO-1
Análisis Estadístico de Señales Fisiológicas
# Análisis Estadístico de la Señal

Este proyecto realiza un análisis detallado de una señal fisiológica (ECG) utilizando técnicas estadísticas descriptivas y modelos de ruido, con el objetivo de evaluar características esenciales de la señal y los efectos del ruido en su comportamiento. Se analizan estadísticos básicos, distribuciones de probabilidad y la resistencia de la señal frente a diferentes tipos de ruido, evaluando su relación señal-ruido (SNR).

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
Se utiliza `wfdb.rdrecord` para cargar una señal fisiológica (ECG) desde un archivo estándar en formato WFDB. En este caso, se seleccionan los primeros 900 puntos de la señal. Este paso inicial permite trabajar con un subconjunto significativo de datos para realizar análisis detallados.

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
<p align="center">
    <img src="https://i.postimg.cc/50qyPvY9/histograma.png" alt="histograma" width="450">
</p>

Se genera un histograma que describe la distribución de los valores de la señal:
- **Frecuencia Absoluta:** La cantidad de veces que aparecen los valores dentro de cada rango.
- **Densidad de Probabilidad (KDE):** Representa de forma suavizada cómo se distribuyen los datos.

**Análisis:**
El histograma revela que los valores de la señal están centrados alrededor de la media y decrecen hacia los extremos. Esto es típico de señales fisiológicas donde la actividad normal se mantiene en un rango definido.

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
<p align="center">
    <img src="https://github.com/user-attachments/assets/d8104ccb-6b13-49c2-b510-abae7d5338f3" alt="image" width="500">
</p>


La gráfica muestra la variación de la señal ECG en el tiempo:
- **Patrones Visuales:** Se pueden observar las ondas características (P, QRS, T) típicas de un ECG.
- **Amplitud:** Indica la intensidad de las variaciones del voltaje.

**Análisis:**
Esta gráfica ayuda a identificar irregularidades o anomalías que podrían ser indicativas de problemas cardiacos.

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
Se calculan los siguientes estadísticos:
- **Media (μ):** Valor promedio de la señal.
- **Desviación Estándar (σ):** Medida de la dispersión de los datos respecto a la media.
- **Coeficiente de Variación (CV):** Relación entre desviación estándar y media, expresada en porcentaje.

$$
\mu = \frac{\sum x_i}{n}, \quad
\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{n-1}}, \quad
CV = \frac{\sigma}{\mu}
$$


**Resultados:**
- Media: -0.0124
- Desviación estándar: 0.131
- Coeficiente de variación: -10.557

**Interpretación:**
La media cercana a cero indica una señal centrada, mientras que el coeficiente de variación muestra una variabilidad moderada.

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
Se obtienen los mismos resultados de manera más eficiente utilizando NumPy.

**Resultados:**
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
$$
P(v) = \frac{\text{Frecuencia Absoluta de } v}{\text{Total de Valores}}
$$

Se calcula la probabilidad de ocurrencia de cada valor único en la señal. Esto ayuda a comprender cómo se distribuyen los valores específicos.

**Ejemplo de Resultados:**
- Valor: -0.28000, Probabilidad: 0.00050
- Valor: 0.00000, Probabilidad: 0.01650

**Análisis:**
La mayoría de los valores tienen baja probabilidad individual, lo que refleja la variabilidad natural de la señal.

---

### 6. Ruido Añadido y Cálculo de SNR
#### 6.1. Ruido Gaussiano
```python
ruido = np.random.normal(0, 0.1, t)
señal_ruidosa = señal + ruido
```
Se añade ruido con distribución normal para simular interferencias aleatorias.

#### 6.2. Ruido de Impulso
```python
prob_impulso = 0.08
impulsos = np.random.choice([0, 1], size=len(señal), p=[1-prob_impulso, prob_impulso])
amplitud_impulso = np.random.choice([-1, 1], size=len(señal)) * 0.2
ruido2 = impulsos * amplitud_impulso
```
Se introducen pulsos aleatorios para simular eventos transitorios abruptos.

#### 6.3. Ruido Tipo Artefacto
```python
prob_imp = 0.15
impul = np.random.choice([0, 1], size=len(señal), p=[1-prob_imp, prob_imp])
amplitud = np.random.choice([-1, 1], size=len(señal)) * 0.2
ruido3 = impul * amplitud
```
Se simulan eventos anómalos de alta amplitud.

#### Cálculo del SNR
El SNR o la Relación Señal-Ruido es una medida que compara el nivel de la señal útil con el nivel del ruido no deseado. En otras palabras, es una forma de medir qué tan clara es una señal en comparación con el ruido que la acompaña. Un SNR alto significa que la señal es mucho más fuerte que el ruido, lo que generalmente resulta en una mejor calidad de la señal. Por otro lado, un SNR bajo indica que el ruido predomina sobre la señal, lo que puede causar distorsión o errores.

$$
\text{SNR (dB)} = 10 \cdot \log_{10} \left( \frac{P_{\text{señal}}}{P_{\text{ruido}}} \right)
$$

```python
def snr(s,r):
    potencia_señal = np.mean(s**2)
    potencia_ruido = np.mean(r**2)
    
    if potencia_ruido == 0:
        return np.inf
    snr = 10 * np.log10(potencia_señal/potencia_ruido) 
    return snr
```
**Señales con Ruido:**
<p align="center">
    <img src="https://github.com/user-attachments/assets/1f1d2a8e-0e72-49a7-9e39-701a1fda1e9f" alt="image" width="500">
</p>
<p align="center">
    <img src="https://github.com/user-attachments/assets/776a5151-430b-4e1d-b91e-a53ff1d90979" alt="image" width="500">
</p>
<p align="center">
    <img src="https://github.com/user-attachments/assets/6ed6d849-933b-41c5-8693-b9a34e554b67" alt="image" width="500">
</p>

**Señales con Ruido Amplificado:**
<p align="center">
    <img src="https://github.com/user-attachments/assets/924d3f6d-c0eb-4e05-a678-a451cb81b9d4" alt="image" width="500">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/1183d8bd-2dd4-40ef-93bb-302e7420e4d9" alt="image" width="500">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/080b90b3-ae34-447e-b9be-ac5e3a59199f" alt="image" width="500">
</p>





**Resultados SNR:**
- **Ruido Gaussiano:** 10.419 dB
- **Ruido Impulso:** 7.274 dB
- **Ruido Artefacto:** 4.066 dB

**Análisis:**
El SNR más alto indica que la señal es más resistente al ruido gaussiano en comparación con el ruido de impulso o artefactos.

---

### 7. Visualización de Ruido
Se grafican las señales contaminadas con ruido:
```python
plt.figure()
plt.plot(señal, label='Señal original')
plt.plot(señal_ruidosa, label='Ruido gaussiano')
plt.legend()
plt.show()
```

**Análisis Visual:**
Las gráficas muestran cómo diferentes tipos de ruido afectan la forma de la señal, con el ruido de artefacto generando mayores distorsiones.

---

## Resultado Esperado
- Histogramas y estadísticas descriptivas.
- Distribución de probabilidad de valores.
- Gráficas de señales con diferentes tipos de ruido.
- Cálculo de SNR para evaluar la calidad de la señal bajo ruido.

## Instrucciones
1. Descargar la señal desde bases de datos como PhysioNet.
2. Ejecutar el código en un entorno Python.
3. Subir este análisis a GitHub con el archivo `README.md` y las gráficas generadas como soporte visual.

