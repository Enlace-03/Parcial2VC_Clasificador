# import cv2, os

# def cargar_imagenes_carpeta(carpeta):
#     imagenes = [] # Lista para almacenar las imágenes cargadas

#     # Recorrer todos los archivos en la carpeta
#     for nombre_archivo in os.listdir(carpeta):
#         ruta_archivo = os.path.join(carpeta, nombre_archivo)# Leer la imagen
#         imagen = cv2.imread(ruta_archivo) # Leer la imagen
#         # Verificar si la imagen se cargó correctamente
#         if imagen is not None:
#             imagenes.append(imagen)
#         else:
#             print(f"No se pudo cargar la imagen: {ruta_archivo}")
#     return imagenes

# def calcular_hog(imagen, pixeles_por_celda=8, celdas_por_bloque=2, bins=9):
#     # Definir los parámetros del HOG
#     hog = cv2.HOGDescriptor(
#         _winSize=(imagen.shape[1] // pixeles_por_celda * pixeles_por_celda,
#                   imagen.shape[0] // pixeles_por_celda * pixeles_por_celda),
#         _blockSize=(celdas_por_bloque * pixeles_por_celda,
#                     celdas_por_bloque * pixeles_por_celda),
#         _blockStride=(pixeles_por_celda, pixeles_por_celda),
#         _cellSize=(pixeles_por_celda, pixeles_por_celda),
#         _nbins=bins
#     )
#     # Calcular el descriptor HOG
#     vector_hog = hog.compute(imagen)
#     return vector_hog.ravel()

# def calcular_fondo(ruta):

#     # Crear el objeto de sustracción de fondo
#     bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
#     frame = cargar_imagenes_carpeta(ruta)

#     for i in range(len(frame)):
#         foreground_mask = bg_subtractor.apply(frame[i])

#     # Obtener el modelo de fondo actual
#     background_model = bg_subtractor.getBackgroundImage()
#     return background_model

# def HOG(ruta):
#     current_frame = cv2.imread(ruta)

#     # Hacer y convertir la diferencia a escala de grises
#     diferencia = cv2.absdiff(current_frame, background_model)
#     diferencia_gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)

#     # Aplicar un umbral para obtener una máscara binaria
#     _, foreground_mask = cv2.threshold(diferencia_gris, 50, 255, cv2.THRESH_BINARY)
#     #Pasar a color nuevamente
#     foreground_mask_color = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)

#     # Aplicar la máscara a la imagen original
#     imagen_resultante = cv2.bitwise_and(current_frame, foreground_mask_color)

#     imagen_resultante_gris = cv2.cvtColor(imagen_resultante, cv2.COLOR_BGR2GRAY)
#     vector_hog = calcular_hog(imagen_resultante_gris)

#     return vector_hog

# #Ejemplo de uso, si se va a usar en este archi descomentar las siguiente lineas a partir de ruta = "./BaseDatos/Positivo/frameb3_0217.jpg"
# #Esto solo se hace una vez, es para calcular el fondo
# carpeta_imagenes_fondo = "./SoloFondo_Entrenamiento" #ruta del archivo
# background_model = calcular_fondo(carpeta_imagenes_fondo)   ##Decirle a romero que esto si o si debe estar descomentado o geenrarlo en otro archivo y mandarlo como entrada de la funcion
# print("Finalizado background_model")

# #Esto se hace para calcular el HOG de cada imagen, es decir esto se itera
# # ruta = "./BaseDatos/Positivo/frameb3_0217.jpg"
# # vector_hog = HOG(ruta)
# # print(len(vector_hog))

import cv2, os
import numpy as np

def cargar_imagenes_carpeta(carpeta):
    imagenes = [] # Lista para almacenar las imágenes cargadas

    # Recorrer todos los archivos en la carpeta
    for nombre_archivo in os.listdir(carpeta):
        ruta_archivo = os.path.join(carpeta, nombre_archivo)# Leer la imagen
        imagen = cv2.imread(ruta_archivo) # Leer la imagen
        # Verificar si la imagen se cargó correctamente
        if imagen is not None:
            imagenes.append(imagen)
        else:
            print(f"No se pudo cargar la imagen: {ruta_archivo}")
    return imagenes

def calcular_hog(imagen, pixeles_por_celda=8, celdas_por_bloque=2, bins=9):
    # Definir los parámetros del HOG
    hog = cv2.HOGDescriptor(
        _winSize=(imagen.shape[1] // pixeles_por_celda * pixeles_por_celda,
                  imagen.shape[0] // pixeles_por_celda * pixeles_por_celda),
        _blockSize=(celdas_por_bloque * pixeles_por_celda,
                    celdas_por_bloque * pixeles_por_celda),
        _blockStride=(pixeles_por_celda, pixeles_por_celda),
        _cellSize=(pixeles_por_celda, pixeles_por_celda),
        _nbins=bins
    )
    # Calcular el descriptor HOG
    vector_hog = hog.compute(imagen)
    return vector_hog.ravel()

def alinear_imagen(imagen_ref, imagen):
    gris_ref = cv2.cvtColor(imagen_ref, cv2.COLOR_BGR2GRAY)
    gris_img = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Detectar puntos clave y descriptores con ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gris_ref, None)
    kp2, des2 = orb.detectAndCompute(gris_img, None)

    # Coincidencia de puntos clave con el método de fuerza bruta
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Ordenar coincidencias por distancia
    matches = sorted(matches, key=lambda x: x.distance)

    # Extraer las coordenadas de los puntos clave de las mejores coincidencias
    puntos_ref = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    puntos_img = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calcular la homografía si hay suficientes puntos
    if len(puntos_ref) >= 4:
        H, _ = cv2.findHomography(puntos_img, puntos_ref, cv2.RANSAC, 5.0)
        # Aplicar la transformación a la imagen
        altura, ancho = imagen_ref.shape[:2]
        imagen_alineada = cv2.warpPerspective(imagen, H, (ancho, altura))
        return imagen_alineada
    else:
        print("No se encontraron suficientes coincidencias para alinear la imagen.")
        return imagen  # Devuelve la imagen sin cambios si no se puede alinear

def calcular_fondo(ruta):

    # Crear el objeto de sustracción de fondo
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)
    frame = cargar_imagenes_carpeta(ruta)

    for i in range(len(frame)):
        nuevo_frame = alinear_imagen(frame[0], frame[i])
        foreground_mask = bg_subtractor.apply(nuevo_frame)

    # Obtener el modelo de fondo actual
    background_model = bg_subtractor.getBackgroundImage()
    return background_model, frame[0]

def HOG(ruta, background_model, imagen_ref):
    frame = cv2.imread(ruta)
    current_frame = alinear_imagen(imagen_ref, frame)

    # Hacer y convertir la diferencia a escala de grises
    diferencia = cv2.absdiff(current_frame, background_model)
    diferencia_gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para obtener una máscara binaria
    _, foreground_mask = cv2.threshold(diferencia_gris, 50, 255, cv2.THRESH_BINARY)
    #Pasar a color nuevamente
    foreground_mask_color = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)

    # cv2.imwrite("foreground_mask.jpg", foreground_mask)
    # cv2.imshow("Fondo Promedio", foreground_mask)
    # cv2.waitKey(0)

    # Aplicar la máscara a la imagen original
    imagen_resultante = cv2.bitwise_and(current_frame, foreground_mask_color)

    imagen_resultante_gris = cv2.cvtColor(imagen_resultante, cv2.COLOR_BGR2GRAY)
    vector_hog = calcular_hog(imagen_resultante_gris)

    # cv2.imwrite("Imagen_final.jpg", imagen_resultante)
    # cv2.imshow("Fondo Promedio", imagen_resultante)
    # cv2.waitKey(0)

    return vector_hog

#Ejemplo de uso
#Esto solo se hace una vez, es para calcular el fondo
carpeta_imagenes_fondo = "./BaseDatos/Negativo/SoloFondo" #ruta del archivo
imagen_ref, background_model = calcular_fondo(carpeta_imagenes_fondo)

# cv2.imwrite("background_model.jpg", background_model)
# cv2.imshow("Fondo Promedio", background_model)
# cv2.waitKey(0)

#Esto se hace para calcular el HOG de cada imagen, es decir esto se itera
ruta = "./BaseDatos/Positivo/frameb1_0109.jpg"
vector_hog = HOG(ruta, background_model, imagen_ref)
print(len(vector_hog))
