import cv2
import os
from Background import HOG
import joblib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

def procesar_video(frames_dir, output_video, clf, frame_width=320, frame_height=180, fps=30, num_frames=605):
    """
    Procesa una serie de frames para detectar objetos/personas y genera un video con los resultados.

    Parámetros:
        frames_dir (str): Directorio donde se encuentran los frames.
        output_video (str): Ruta donde se guardará el video procesado.
        clf (sklearn.svm.SVC): Modelo preentrenado para hacer las predicciones.
        frame_width (int): Ancho de los frames procesados.
        frame_height (int): Alto de los frames procesados.
        fps (int): Fotogramas por segundo del video de salida.
        num_frames (int): Número total de frames a procesar.
    """
    # Inicializar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Procesar cada frame
    for i in range(num_frames):
        frame_filename = f"resized_frame_{i:04d}.jpg"  # Formato de nombre de archivo
        frame_path = os.path.join(frames_dir, frame_filename)

        if not os.path.exists(frame_path):
            print(f"Frame no encontrado: {frame_path}")
            continue

        # Leer el frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"No se pudo cargar el frame: {frame_path}")
            continue

        # Calcular el vector HOG para el frame
        vector_hog = HOG(frame_path)

        # Hacer la predicción
        prediccion = clf.predict([vector_hog])

        # Dibujar un bombillito si hay una persona
        if prediccion[0] == 1:  # Si hay una persona
            cv2.circle(frame, (frame_width - 30, frame_height - 30), 20, (0, 255, 255), -1)  # Bombillito amarillo

        # Escribir el frame procesado al video
        video_writer.write(frame)

    # Liberar el escritor de video
    video_writer.release()
    print(f"Video procesado guardado en: {output_video}")


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


#Definir las variables de entrada para procesar el video:
frames_dir = "framesPruebaPrediccion"   #Nombre de la carpeta en que están los frames
output_video = "video_procesado.mp4"    #Nombre del video de salida que reconoce a las personas
frame_width = 320                       #Ancho de los frames
frame_height = 180                      #Largo de los frames
fps = 30                                #Fotogramas por segundo
num_frames = 605                        #Número de frames


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

# Procesar el video
procesar_video(frames_dir, output_video, clf, frame_width, frame_height, fps, num_frames)

