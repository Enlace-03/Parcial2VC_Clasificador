from Background import calcular_fondo, HOG
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

# Cargar X e Y
X = np.load('XCaracteristicas_Background_HOG_BaseActualizadaAlineado.npy')
Y = np.load('YCaracteristicas_Background_HOG_BaseActualizadaAlineado.npy')

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

# Intentar cargar el modelo, si no existe entrenarlo
model_filename = "modelo_svm_hog_Alineado.pkl"

try:
    clf = joblib.load(model_filename)
    print("Modelo cargado correctamente")
except FileNotFoundError:
    print("Modelo no encontrado, entrenando uno nuevo...")
    clf = SVC(kernel='rbf', C=1)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_filename)
    print(f"Modelo guardado en {model_filename}")

## Predicción y evaluación
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")




#######################################################################################################

# #Probar modelo con imagenes que no estan en la base de datos
# #Esto solo se hace una vez, es para calcular el fondo
# carpeta_imagenes_fondo = "./BaseDatos/Negativo/SoloFondo_Entrenamiento" #ruta del archivo
# imagen_ref, background_model = calcular_fondo(carpeta_imagenes_fondo)
#
#
#
#
# #ruta = "./PruebaFuncionamiento/Kevin.jpg"
# #ruta = "./PruebaFuncionamiento/solofondo3.jpg"
# #ruta = "./BaseDatos/Negativo/SoloFondo/frameb1_0250.jpg"
# #ruta = "./framesPruebaPrediccion/resized_frame_0213.jpg"#Ruta con una persona
# ruta = "./framesPruebaPrediccion/resized_frame_0252.jpg" #Ruta con fondo con objetos
# #ruta = "./framesPruebaPrediccion/resized_frame_0001.jpg" #Rura de fondo solo
#
# vector_hog = HOG(ruta, background_model, imagen_ref)
#
# # Hacer la predicción
# prediccion = clf.predict([vector_hog])
#
#
# print("Prediccion etiqueta: ", prediccion[0])
# # Mostrar la predicción
# print("Predicción para la nueva imagen:", "Positivo" if prediccion[0] == 1 else "Negativo")

