#Se hace el entrenamiento del SVC con algoritmo usando open cv basado en la sustraccion del fondo con metodo gaussiano iterable
#Al tener el fondo se aplica un enmascarmaiento a la imgen x y el resultado se le aplica HOG que se usa como caracteristica

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Background import calcular_fondo, HOG



# Inicializar listas para las características y etiquetas
Xp = []  # Características HOG
y = []  # Etiquetas (1 para positivo, 0 para negativo)

# # Ruta de las bases de datos
ruta_positivos = './BaseDatos/Positivo_Entrenamiento/'
ruta_negativos = './BaseDatos/Negativo/SoloFondo_Entrenamiento/'
ruta_negativos2 = './BaseDatos/Negativo/ConObjetos_Entrenamiento/'


# Cargar imágenes negativas con objetos
for filename in os.listdir(ruta_negativos2):
    ruta_imagen = os.path.join(ruta_negativos2, filename)
    vector_hog = HOG(ruta_imagen)
    Xp.append(vector_hog)
    y.append(0)  # 0 para imágenes sin persona
    print("x agregado y agregado negativo con objetos")

# Cargar imágenes negativas sin objetos
for filename in os.listdir(ruta_negativos):
    ruta_imagen = os.path.join(ruta_negativos, filename)
    vector_hog = HOG(ruta_imagen)
    Xp.append(vector_hog)
    y.append(0)  # 0 para imágenes sin persona
    print("x agregado y agregado negativo SIN objetos")

# Cargar imágenes positivas
for filename in os.listdir(ruta_positivos):
    ruta_imagen = os.path.join(ruta_positivos, filename)
    vector_hog = HOG(ruta_imagen)
    Xp.append(vector_hog)
    y.append(1)  # 1 para imágenes con persona
    print("x agregado y positivos")


# Convertir las listas a arrays de numpy
X = np.array(Xp)
Y = np.array(y)


np.save('XCaracteristicas_Background_HOG_BaseActualizada.npy', X)
np.save('YCaracteristicas_Background_HOG_BaseActualizada.npy',Y)
# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

# Crear el clasificador SVC
clf = SVC(kernel='rbf', C=1)  #

# Entrenar el modelo
clf.fit(X_train, y_train)

# Predecir con los datos de prueba
y_pred = clf.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")