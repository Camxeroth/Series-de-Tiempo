# Predicción de Series de Tiempo con Redes Neuronales LSTM
### Modelamiento con Aprendizaje Profundo — TensorFlow / Keras

**Asignatura:** Modelamiento — Actividad Autónoma 6  
**Unidad:** 3 — Series de Tiempo  
**Tema:** Redes Neuronales Recurrentes (LSTM) para Pronóstico Temporal  
**Entorno:** Python 3 · Jupyter Notebook · Google Colab

---

## Descripción General

Este repositorio implementa un modelo de pronóstico de series temporales basado en redes neuronales recurrentes de memoria a largo y corto plazo (**LSTM**, por sus siglas en inglés: *Long Short-Term Memory*). El conjunto de datos empleado registra la **producción mensual de leche en libras** (`monthly-milk-production-pounds.csv`), con una partición temporal estricta para evaluar la capacidad predictiva del modelo fuera de muestra.

El enfoque combina preprocesamiento estadístico clásico (descomposición estacional, normalización) con arquitecturas de aprendizaje profundo orientadas a la modelación de dependencias temporales de largo alcance.

---

## Conjunto de Datos

| Atributo | Detalle |
|---|---|
| Archivo | `monthly-milk-production-pounds.csv` |
| Variable objetivo | Producción mensual de leche (en libras) |
| Índice temporal | Mensual (formato fecha) |
| Partición entrenamiento | Hasta diciembre de 1974 |
| Partición prueba | Enero de 1975 en adelante |

---

## Metodología

El flujo de trabajo sigue una cadena secuencial de transformaciones que va desde la ingesta del dato crudo hasta la evaluación cuantitativa del pronóstico:

```
Dato crudo (CSV)
    → Análisis descriptivo y descomposición estacional (STL)
    → Autocorrelación y diagnóstico de dependencia temporal
    → Partición train / test
    → Normalización MinMaxScaler [0, 1]
    → Construcción de ventanas deslizantes (LAG = 12)
    → Redimensionamiento tensorial (batch, timesteps, features)
    → Definición de arquitectura LSTM apilada
    → Entrenamiento con parada temprana (EarlyStopping)
    → Evaluación de métricas (RMSE, R², MAPE)
    → Visualización de resultados
```

---

## Hiperparámetros del Modelo

| Parámetro | Valor |
|---|---|
| Retardos temporales (`LAG`) | 12 meses |
| Neuronas — capa 1 (`HIDDEN1`) | 32 |
| Neuronas — capa 2 (`HIDDEN2`) | 16 |
| Neuronas — capa 3 (`HIDDEN3`) | 8 |
| Épocas máximas (`EPOCHS`) | 150 |
| Tamaño de lote (*batch size*) | 32 |
| Fracción de validación | 20 % del conjunto de entrenamiento |
| Parada temprana (`patience`) | 10 épocas sin mejora en `loss` |

---

## Arquitectura de la Red Neuronal

El modelo consiste en una red neuronal recurrente profunda con tres capas LSTM apiladas, regularización mediante abandono (*Dropout*) y una capa densa de salida:

```
Capa de entrada   →  Tensor 3D (batch_size, 12 timesteps, 1 feature)
        ↓
LSTM(32 unidades, activation='relu', return_sequences=True)
        ↓
Dropout(0.2)
        ↓
LSTM(16 unidades, return_sequences=True)
        ↓
Dropout(0.2)
        ↓
LSTM(8 unidades)
        ↓
Dropout(0.2)
        ↓
Dense(1 unidad)   →  Salida escalar normalizada
```

**Función de pérdida:** Error cuadrático medio (`mean_squared_error`)  
**Optimizador:** Adam

---

## Preprocesamiento

### Descomposición Estacional

La serie original fue descompuesta mediante el modelo aditivo (`seasonal_decompose`, `statsmodels`) para aislar los siguientes componentes estructurales:

- **Tendencia** — evolución de largo plazo de la producción.
- **Estacionalidad** — ciclo anual recurrente con periodicidad mensual.
- **Residuo** — variabilidad no explicada por los componentes anteriores.

La elección del modelo aditivo se justifica porque la amplitud del componente estacional permanece aproximadamente constante en relación al nivel de la tendencia.

### Normalización

Todos los valores fueron escalados al intervalo [0, 1] mediante `MinMaxScaler`, ajustando el transformador exclusivamente sobre el conjunto de entrenamiento y aplicándolo al conjunto de prueba, evitando así la fuga de información (*data leakage*).

### Ventanas Deslizantes

La función `window(ds, lags)` transforma la serie univariada en una matriz supervisada con estructura de retardos. Cada observación en el instante `t` queda asociada a sus 12 valores anteriores como variables predictoras (`X`) y al valor en `t+1` como variable objetivo (`y`).

---

## Estructura del Notebook

| Sección | Contenido |
|---|---|
| 1.1 | Configuración del entorno TensorFlow y carga de librerías |
| Descripción gráfica | Visualización temporal de la serie completa |
| Descomposición STL | Aislamiento de tendencia, estacionalidad y residuo |
| Autocorrelación | Diagnóstico de dependencia temporal |
| Partición train/test | División temporal estricta de los datos |
| Normalización | Escalado MinMax y construcción de ventanas deslizantes |
| Arquitectura LSTM | Definición, compilación y entrenamiento del modelo |
| Evaluación | Cálculo de RMSE, R² y MAPE en train y test |
| Visualización | Comparación gráfica entre predicción y valores reales |
| 1.3 | Análisis de mejoras potenciales al modelo |

---

## Métricas de Evaluación

El modelo es evaluado mediante tres métricas complementarias, calculadas tanto sobre el conjunto de entrenamiento como sobre el de prueba:

| Métrica | Descripción |
|---|---|
| **RMSE** | Raíz del error cuadrático medio — penaliza errores grandes |
| **R²** | Coeficiente de determinación — proporción de varianza explicada |
| **MAPE** | Error porcentual absoluto medio — interpretable en escala relativa |

---

## Análisis de Mejoras (Ejercicio 1.3)

El notebook identifica las siguientes líneas de mejora para incrementar la precisión predictiva del modelo:

**Arquitectura de la red.** Incorporar capas LSTM adicionales con mayor número de unidades ocultas puede ampliar la capacidad representacional del modelo, siempre bajo supervisión del error de validación para evitar sobreajuste (*overfitting*).

**Épocas de entrenamiento.** Aumentar el número máximo de épocas, en combinación con la parada temprana ya implementada, permite que el modelo converja a soluciones más refinadas sin riesgo de degradación por entrenamiento excesivo.

**Transformaciones de datos.** Explorar diferenciación temporal o transformaciones logarítmicas puede estabilizar la varianza y facilitar el aprendizaje de patrones de largo plazo.

**Tamaño de la ventana temporal.** Ajustar el parámetro `LAG` a valores superiores o inferiores a 12 permite evaluar si el modelo captura mejor las dependencias con horizontes temporales distintos.

**Reducción del conjunto de entrenamiento.** Limitar los datos de entrenamiento a períodos más recientes puede mejorar la generalización si la dinámica de la serie ha cambiado estructuralmente a lo largo del tiempo.

---

## Dependencias

```python
tensorflow >= 2.x
pandas
numpy
scikit-learn
matplotlib
statsmodels
math
datetime
```

Para ejecutar el notebook en Google Colab no es necesaria ninguna instalación adicional. En entorno local se recomienda crear un entorno virtual e instalar las dependencias con:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib statsmodels
```

---

## Estructura del Repositorio

```
.
├── data/
│   └── monthly-milk-production-pounds.csv
├── modelamiento_A_AA6.ipynb
└── README.md
```

---

## Licencia

Este trabajo fue desarrollado en el marco académico de la asignatura de Modelamiento, Facultad de Ingeniería. Todos los derechos reservados por el autor.
