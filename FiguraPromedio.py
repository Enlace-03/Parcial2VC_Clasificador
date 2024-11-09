#Tener en cuenta para correr el archivo

#1. Descargar la carpeta de drive y gyardarla en la carpeta principal del proyecto
#2. Guardarla como SoloFondo, en su defecto cambiar la ruta en el codigo
#3. Al correr el codigo se genera una imagen llamada fondo_promedio.jpg
#   eliminarla antes de volver a correrlo una vez genrada esta por primera vez.
#4. El codigo comentado es la imagen promedio anterior, sin homografia. Esto se
#   se puede usar para la presentación y hacer comparación.

import os
import cv2
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

def calcular_fondo_promedio(imagenes):
    # Utilizar la primera imagen como referencia para la alineación
    imagen_ref = imagenes[0]
    suma_imagenes = np.zeros_like(imagen_ref, dtype=np.float32)

    # Alinear y acumular cada imagen
    for imagen in imagenes:
        imagen_alineada = alinear_imagen(imagen_ref, imagen)
        suma_imagenes += imagen_alineada.astype(np.float32)

    # Calcular el promedio y convertir a uint8
    fondo_promedio = suma_imagenes / len(imagenes)
    fondo_promedio = fondo_promedio.astype(np.uint8)
    return fondo_promedio

carpeta_imagenes_fondo = "./SoloFondo" #ruta del archivo
imagenes_fondo = cargar_imagenes_carpeta(carpeta_imagenes_fondo) #se cargan las imagenes
print(f"Se cargaron {len(imagenes_fondo)} imágenes de la carpeta '{carpeta_imagenes_fondo}'")

fondo_promedio = calcular_fondo_promedio(imagenes_fondo) #promedio
cv2.imwrite("fondo_promedio.jpg", fondo_promedio) #se guarda una imagen

# Mostrar la imagen del fondo promedio
cv2.imshow("Fondo Promedio", fondo_promedio)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import os
# import cv2
# import numpy as np

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

# def calcular_fondo_promedio(imagenes):
#     suma_imagenes = np.zeros_like(imagenes[0], dtype=np.float32)# Inicializar una matriz para acumular las imágenes

#     # Sumar todas las imágenes en la lista
#     for imagen in imagenes:
#         suma_imagenes += imagen.astype(np.float32)

#     fondo_promedio = suma_imagenes / len(imagenes)
#     fondo_promedio = fondo_promedio.astype(np.uint8)# Convertir el fondo promedio a uint8 para que sea una imagen válida
#     return fondo_promedio

# carpeta_imagenes_fondo = "./SoloFondo" #ruta del archivo
# imagenes_fondo = cargar_imagenes_carpeta(carpeta_imagenes_fondo) #se cargan las imagenes
# print(f"Se cargaron {len(imagenes_fondo)} imágenes de la carpeta '{carpeta_imagenes_fondo}'")

# fondo_promedio = calcular_fondo_promedio(imagenes_fondo) #promedio
# cv2.imwrite("fondo_promedio.jpg", fondo_promedio) #se guarda una imagen

# # Mostrar la imagen del fondo promedio
# cv2.imshow("Fondo Promedio", fondo_promedio)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
