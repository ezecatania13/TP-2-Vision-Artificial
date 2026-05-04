# Proyecto: Clasificación de Figuras con Machine Learning

Este proyecto utiliza visión artificial y aprendizaje automático para reconocer y clasificar figuras geométricas simples (como rectángulos, triángulos y círculos) a partir de la imagen de una webcam. El sistema reemplaza la función clásica de comparación de formas (`matchShapes`) por un modelo de machine learning entrenado con descriptores de contornos (invariantes de Hu).

## Estructura del Proyecto

- **generador_descriptores.py**: Captura video de la webcam, detecta contornos y, al presionar ESPACIO, calcula los 7 invariantes de Hu del contorno más grande. Guarda cada muestra en un archivo CSV junto con la etiqueta de la figura.
- **entrenador.py**: Lee el archivo CSV generado, entrena un árbol de decisión con scikit-learn usando los invariantes de Hu y las etiquetas, y guarda el modelo entrenado en un archivo (`modelo.joblib`).
- **clasificador.py**: Usa la webcam para detectar contornos en tiempo real, calcula los invariantes de Hu y predice la clase de la figura usando el modelo entrenado. Muestra la predicción en pantalla.
- **invariantes_hu.csv**: Archivo generado automáticamente por el generador de descriptores, contiene los descriptores y etiquetas para el entrenamiento.
- **modelo.joblib**: Archivo generado por el entrenador, contiene el modelo de machine learning entrenado.

## Requisitos de Instalación

- Python 3.7 o superior
- Paquetes:
  - opencv-python
  - numpy
  - scikit-learn
  - joblib
  - matplotlib (opcional, para visualizar el árbol de decisión)

Instala los paquetes necesarios con:
```
pip install opencv-python numpy scikit-learn joblib matplotlib
```

## Guía de Ejecución

1. **Generar descriptores:**
   - Ejecuta `generador_descriptores.py`:
     ```
     python generador_descriptores.py
     ```
   - Coloca una figura sobre un fondo liso y:
     - Presiona **ESPACIO** para mostrar los invariantes de Hu del contorno más grande sin guardarlos (útil si te equivocaste o no quieres guardar esa muestra).
     - Presiona **ENTER** para guardar los invariantes de Hu y la etiqueta en el archivo CSV. Ingresa la etiqueta numérica correspondiente (ejemplo: 1 para rectángulo, 2 para triángulo, etc.).
     - Presiona **ESC** para salir.
   - Repite para varias muestras y figuras.

2. **Entrenar el modelo:**
   - Ejecuta `entrenador.py`:
     ```
     python entrenador.py
     ```
   - Esto entrenará el modelo y lo guardará en `modelo.joblib`. Se mostrará una visualización del árbol de decisión.

3. **Clasificar en tiempo real:**
   - Ejecuta `clasificador.py`:
     ```
     python clasificador.py
     ```
   - El sistema detectará figuras en la webcam y mostrará la predicción en pantalla.

## Notas
- Usa siempre las mismas etiquetas numéricas para cada tipo de figura en todo el proceso.
- Para mejores resultados, usa fondos lisos y buena iluminación.
- Si necesitas agregar nuevas figuras, actualiza el diccionario de etiquetas en `clasificador.py`.

---

Proyecto realizado para la materia Visión Artificial, UNLAM, 2026.
