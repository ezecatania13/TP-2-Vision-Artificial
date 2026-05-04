import csv
from sklearn import tree
from joblib import dump
import numpy as np

# Nombre del archivo CSV generado por el generador de descriptores
CSV_FILENAME = "invariantes_hu.csv"
# Nombre del archivo donde se guardará el modelo entrenado
MODEL_FILENAME = "modelo.joblib"

# Leer el dataset desde el archivo CSV
X = []  # Descriptores (invariantes de Hu)
Y = []  # Etiquetas

with open(CSV_FILENAME, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Salta el encabezado
    for row in reader:
        # Los primeros 7 valores son los invariantes de Hu, el último es la etiqueta
        X.append([float(val) for val in row[:7]])
        Y.append(int(row[7]))


# Mezclar aleatoriamente las filas del dataset
from sklearn.utils import shuffle
X, Y = shuffle(np.array(X), np.array(Y), random_state=42)

# Entrenar el árbol de decisión
clasificador = tree.DecisionTreeClassifier()
clasificador.fit(X, Y)

# Guardar el modelo entrenado
dump(clasificador, MODEL_FILENAME)
print(f"Modelo entrenado y guardado en {MODEL_FILENAME}")

# (Opcional) Visualizar el árbol de decisión
import matplotlib.pyplot as plt
plt.figure(figsize=(12,8))
tree.plot_tree(clasificador, filled=True, feature_names=header[:7], class_names=[str(i) for i in set(Y)])
plt.savefig("arbol_decision.png")
plt.show()
print("Imagen del árbol de decisión guardada como arbol_decision.png")
