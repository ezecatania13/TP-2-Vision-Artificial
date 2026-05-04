import cv2
import numpy as np
from joblib import load

# Diccionario de etiquetas (ajusta según tus clases)
diccionario_etiquetas = {
    1: "Triangulo",
    2: "Rectangulo",
    3: "Circulo"
}

# Carga el modelo entrenado
clasificador = load("modelo.joblib")

cap = cv2.VideoCapture(0)
print("Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 500:  # Filtra contornos pequeños
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            moments = cv2.moments(c)
            hu = cv2.HuMoments(moments).flatten().reshape(1, -1)
            etiqueta_predicha = clasificador.predict(hu)[0]
            # Obtener la probabilidad de predicción
            proba = clasificador.predict_proba(hu)[0]
            idx = list(clasificador.classes_).index(etiqueta_predicha)
            probabilidad = proba[idx]
            texto = f"{diccionario_etiquetas.get(etiqueta_predicha, str(etiqueta_predicha))} ({probabilidad*100:.1f}%)"
            x, y, w, h = cv2.boundingRect(c)
            cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow('Clasificador', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
