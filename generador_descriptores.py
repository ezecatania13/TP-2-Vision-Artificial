
import cv2
import numpy as np
import csv
import os

# Archivo donde se guardarán los invariantes
CSV_FILENAME = "invariantes_hu.csv"

# Inicializa la cámara
cap = cv2.VideoCapture(0)


print("Presiona ENTER para guardar los invariantes de Hu del contorno más grande.")
print("Presiona ESPACIO para mostrar los invariantes sin guardar.")
print("Presiona ESC para salir.")

# Si el archivo no existe, crea el encabezado
if not os.path.exists(CSV_FILENAME):
    with open(CSV_FILENAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"hu{i+1}" for i in range(7)] + ["etiqueta"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convierte a escala de grises y binariza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Encuentra contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja contornos y selecciona el más grande
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC para salir
        break
    elif key == 32 and contours:  # ESPACIO para mostrar y preguntar si guardar
        moments = cv2.moments(c)
        hu = cv2.HuMoments(moments).flatten()
        print("Invariantes de Hu:", hu)
        respuesta = input("¿Deseas guardar estos invariantes? (s/n): ").strip().lower()
        if respuesta == 's':
            etiqueta = input("Ingresa la etiqueta numérica para este contorno: ")
            with open(CSV_FILENAME, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(hu) + [etiqueta])
            print(f"Guardado en {CSV_FILENAME}")

cap.release()
cv2.destroyAllWindows()
