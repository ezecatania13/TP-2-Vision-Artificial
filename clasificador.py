import cv2
import numpy as np
from joblib import load

# Diccionario que asocia las etiquetas numéricas con nombres de figuras
diccionario_etiquetas = {
    1: "Triangulo",
    2: "Rectangulo",
    3: "Circulo"
}

# Carga el modelo previamente entrenado (árbol de decisión)
clasificador = load("modelo.joblib")

# Inicializa la cámara web (índice 0)
cap = cv2.VideoCapture(0)
print("Presiona ESC para salir.")

# Bucle principal de captura y clasificación
while True:
    ret, frame = cap.read()
    if not ret:
        break # Si no se pudo capturar, termina el bucle

    # Convierte la imagen a escala de grises, aplica umbral para binarizar y encuentra contornos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Procesa cada contorno encontrado
    for c in contours:
        if cv2.contourArea(c) > 500:  # Filtra contornos pequeños (ruido)
            # Dibuja el contorno sobre el frame original
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            # Calcula los momentos y los invariantes de Hu para el contorno
            moments = cv2.moments(c)
            hu = cv2.HuMoments(moments).flatten().reshape(1, -1)
            # Predice la etiqueta de la figura usando el modelo entrenado
            etiqueta_predicha = clasificador.predict(hu)[0]
            texto = diccionario_etiquetas.get(etiqueta_predicha, str(etiqueta_predicha))
            # Obtiene la posición para mostrar el texto
            x, y, w, h = cv2.boundingRect(c)
            cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # Muestra el frame con los resultados
    cv2.imshow('Clasificador', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Si se presiona ESC, termina el bucle
        break

# Libera la cámara y cierra las ventanas
cap.release()
cv2.destroyAllWindows()
