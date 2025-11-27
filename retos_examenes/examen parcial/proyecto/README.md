# 🂡 Sistema de Reconocimiento de Cartas de Póker  
**Visión Artificial sin aprendizaje automático – Procesamiento clásico de imágenes**

Este proyecto implementa un sistema completo para **detectar y reconocer cartas de póker en tiempo real**, cumpliendo estrictamente la condición de **no usar redes neuronales ni modelos entrenados**.  
Todo el reconocimiento se basa únicamente en **procesamiento clásico de imagen, correlación por plantillas y un clasificador k-NN artesanal** (no entrenado con ML).

---

# 📌 Características Principales

### ✔ Detección automática de cartas  
- Segmentación del tapete verde en HSV  
- Detección de contornos  
- Warp perspectiva para extraer la carta enderezada  

### ✔ Reconocimiento de número (Rank)  
- Binarización + normalización  
- Recorte de símbolo principal  
- Reducción dimensional (32×48 px)  
- **Clasificador k-NN implementado manualmente**  
- Sin librerías de ML, solo operaciones matriciales

### ✔ Reconocimiento del palo (Suit)  
- Motor `DetectorPalos` diseñado a medida  
- Detección por forma, color y rotaciones  
- Fallback con *template matching* clásico  
- Correlación TM_CCOEFF_NORMED

### ✔ Soporte para rotaciones  
Cada carta se clasifica en rotaciones:  
**0°, 90°, 180°, 270° → se elige la mejor con su score**

### ✔ Soporte para múltiples cartas simultáneas  
Cada contorno se procesa individualmente.

### ✔ Totalmente compatible con escenario real de examen  
- Cámara cenital  
- Tapete verde  
- Iluminación moderada  
- Varias cartas visibles  
- Distintas orientaciones

---

# 🧩 Estructura del Proyecto

```bash
proyecto/
│
├── plantillas/
│   ├── ranks/                # símbolos sin limpiar
│   ├── ranks_limpios/        # símbolos recortados y binarizados (para k-NN)
│   ├── suits/                # palos originales
│   ├── suits_limpios/        # palos limpios y binarizados
│
├── src/
│   ├── detector_cartas.py    # detección general + clasificación base
│   ├── detector_realtime.py  # ejecución con cámara
│   ├── detector_palos.py     # identificación robusta de palos
│   ├── limpiar_plantillas_ranks.py
│   ├── camara.py
│
├── README.md                 # (este documento)
└── requirements.txt
```

---

## ⚙️ Tecnologías utilizadas

| Componente | Uso |
| :--- | :--- |
| **OpenCV** | Contornos, warping, HSV masking, template matching |
| **NumPy** | Operaciones matriciales, normalización |
| **k-NN manual** | Clasificación de símbolos del rank |
| **Diferencias morfológicas** | Limpieza de máscaras |
| **Matching clásico** | Plantillas para palos |
| **Descriptores simples** | Análisis de color (rojo/negro) |
| **Rotación de carta** | Clasificador robusto a orientación |

---

## 🏗️ Arquitectura del sistema

### 1️⃣ Detección de cartas


Usa:

* Conversión **HSV**
* Segmentación de no-verde
* Operaciones morfológicas
* Contornos externos
* Aprox. poligonal de 4 puntos
* Warping a formato estándar **250×400 px**

Salida por carta:
```json
{
  "quad": puntos_originales,
  "warped": carta_enderezada
}
```

### 2️⃣ Clasificación de Rank (número/letra)

Basado en:

✔️ Binarización adaptativa
✔️ Normalización a 32×48 px
✔️ Extracción del contorno mayor
✔️ k-NN manual con distancia Euclídea

Ventajas:

No depende de orientación (gracias al módulo de rotación)

Muy estable después de limpiar plantillas

### 3️⃣ Clasificación de Palo (suit)

Basada en dos estrategias combinadas:

A. DetectorPalos (forma + color + contorno)

Evalúa:

Geometría del símbolo grande

Simetría izquierda/derecha

Ratio vertical

Color estimado (rojo/negro)

B. Template Matching clásico

Si DetectorPalos < 0.35 de confianza → fallback:

matchTemplate(TM_CCOEFF_NORMED)

### 4️⃣ Robustez a rotación

La carta se clasifica en las orientaciones:

0°
90°
180°
270°


Para cada rotación:

Se extrae ROI rank + ROI suit

Se clasifica

Se evalúa score

Se queda la orientación con mejor suma de scores

#### 🎥 Modo tiempo real

Ejecutado con:

python detector_realtime.py


Incluye:

- Detección de múltiples cartas

- Clasificación en cada rotación

- Visualización en pantalla

- Contornos y texto superpuesto

- Debug opcional

#### 📦 Instalación
```pip
pip install opencv-python numpy
```

- No requiere librerías adicionales.

#### 🚀 Uso básico
🖼️ Reconocimiento en una imagen
* python detector_cartas.py


Ajustar en el archivo:
```python
imagen_prueba = "../imagenes/prueba1.jpg"
```

#### 🎥 Reconocimiento con cámara
* python detector_realtime.py


Presiona Q para salir.

#### 🧪 Resultados esperados

- Precisión general rank: 70–85%

- Precisión general suit: 65–80%

- Detección múltiple: ✔️

- Rotación libre: ✔️

- Examen: reconocimiento de 10 cartas con posiciones aleatorias: ✔️

#### 📝 Justificación técnica
* ✔️ ¿Por qué segmentación por verde?

La forma más robusta de aislar la carta del fondo sin ML.

* ✔️ ¿Por qué warping fijo 250×400?

Permite ROIs relativos y plantillas estables.

* ✔️ ¿Por qué k-NN manual?

Se ajusta a la restricción “sin aprendizaje entrenado”, ya que:

No se entrena nada

Solo compara distancias entre plantillas

Es 100% legal en requisitos.

* ✔️ ¿Por qué matching para palos?

Los palos son más difíciles por:

Variaciones de color

Tamaños distintos

Rotación

Combinar DetectorPalos + fallback asegura estabilidad.

#### 📄 Requisitos del examen (cumplidos)
- Requisito	Estado
- 1 carta totalmente visible	✔️
- Varias cartas	✔️
- Rotaciones libres	✔️
- Distorsión leve	✔️
- Fondo verde	✔️
- Sin redes neuronales	✔️
- Documentación técnica detallada	✔️
- Código limpio	✔️

#### 📚 Autor

Proyecto desarrollado por Cayetano Castillo Ruiz