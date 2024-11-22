from Background import calcular_fondo, HOG
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.svm import SVC
# import numpy as np
#
# from sklearn.model_selection import KFold, cross_val_score
#
# # Cargar X e y desde los archivos guardados
# X = np.load('XCaracteristicas_Background_HOG_BaseActualizada.npy')
# Y = np.load('YCaracteristicas_Background_HOG_BaseActualizada.npy')
#
#
# unique, counts = np.unique(Y, return_counts=True)
# print("Distribución de clases:", dict(zip(unique, counts)))
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)
#
# # Crear el clasificador SVC
# clf = SVC(kernel='rbf', C=1)  # Usar un kernel lineal
#
# # Entrenar el modelo
# clf.fit(X_train, y_train)
#
#
#
# # Predecir con los datos de prueba
# y_pred = clf.predict(X_test)
#
# # Evaluar el modelo
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Precisión del modelo por train_test_split: {accuracy * 100:.2f}%")


import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

# # Cargar X e Y
# X = np.load('XCaracteristicas_Background_HOG_BaseActualizada.npy')
# Y = np.load('YCaracteristicas_Background_HOG_BaseActualizada.npy')
#
# # Dividir los datos
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

# Intentar cargar el modelo, si no existe entrenarlo
model_filename = "modelo_svm_hog.pkl"

try:
    clf = joblib.load(model_filename)
    print("Modelo cargado correctamente")
except FileNotFoundError:
    print("Modelo no encontrado, entrenando uno nuevo...")
    # clf = SVC(kernel='rbf', C=1)
    # clf.fit(X_train, y_train)
    # joblib.dump(clf, model_filename)
    # print(f"Modelo guardado en {model_filename}")

# # Predicción y evaluación
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Precisión del modelo: {accuracy * 100:.2f}%")



#######################################################################################################

#Probar modelo con imagenes que no estan en la base de datos

#ruta = "./PruebaFuncionamiento/Kevin.jpg"
#ruta = "./PruebaFuncionamiento/solofondo3.jpg"
# ruta = "./BaseDatos/Negativo/SoloFondo/frameb1_0250.jpg"
ruta = "./framesPruebaPrediccion/resized_frame_0213.jpg"#Ruta con una persona
#ruta = "./framesPruebaPrediccion/resized_frame_0236.jpg" #Ruta con fondo con objetos
#ruta = "./framesPruebaPrediccion/resized_frame_0001.jpg" #Rura de fondo solo

vector_hog = HOG(ruta)
print(len(vector_hog))


# Hacer la predicción
prediccion = clf.predict([vector_hog])


print("Prediccion etiqueta: ", prediccion[0])
# Mostrar la predicción
print("Predicción para la nueva imagen:", "Positivo" if prediccion[0] == 1 else "Negativo")

