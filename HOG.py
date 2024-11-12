import cv2
import matplotlib.pyplot as plt
import  numpy as np

def alinear_imagen(imagen_ref, imagen):
    gris_ref = imagen_ref
    gris_img = imagen

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

def cargar_y_convertir_a_grises(ruta):
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    return imagen

def sustraccion_fondo(imagen, imagen_fondo):
    diferencia = cv2.absdiff(imagen, imagen_fondo)
    return diferencia

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


# Ejemplo de uso
# Las dos imagenes
ruta_imagen = "./Personas/frameb1_0103.jpg"
ruta_imagen_fondo = "./fondo_promedio.jpg"

# se convierten a escala de grises
imagen_grises = cargar_y_convertir_a_grises(ruta_imagen)
imagen_grises_fondo = cargar_y_convertir_a_grises(ruta_imagen_fondo)

# alinea la nueva imagen con el fondo
image_alineada = alinear_imagen(imagen_grises_fondo, imagen_grises)

#elimina el fondo
imagen_sin_fondo = sustraccion_fondo(image_alineada, imagen_grises_fondo)
vector_hog = calcular_hog(imagen_sin_fondo)

print("Longitud del vector HOG:", len(vector_hog))

# Mostrar la imagen de magnitud de gradiente como ejemplo
plt.imshow(imagen_sin_fondo, cmap="gray")
plt.title("Imagen en Escala de Grises")
plt.axis("off")
plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# def cargar_y_convertir_a_grises(ruta):
#     imagen = cv2.imread(ruta)
#     imagen_grises = 0.2989 * imagen[:, :, 2] + 0.5870 * imagen[:, :, 1] + 0.1140 * imagen[:, :, 0]
#     return imagen_grises

# def calcular_gradientes(imagen):
#     # Definir los kernels de Sobel
#     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

#     # Inicializar las matrices para los gradientes
#     gradiente_x = np.zeros_like(imagen)
#     gradiente_y = np.zeros_like(imagen)

#     # Aplicar convolución manualmente (sin bordes)
#     for i in range(1, imagen.shape[0] - 1):
#         for j in range(1, imagen.shape[1] - 1):
#             # Extraer la región de 3x3
#             region = imagen[i - 1:i + 2, j - 1:j + 2]
#             # Calcular la convolución con los filtros Sobel
#             gradiente_x[i, j] = np.sum(region * sobel_x)
#             gradiente_y[i, j] = np.sum(region * sobel_y)

#     return gradiente_x, gradiente_y

# def calcular_magnitud_orientacion(gradiente_x, gradiente_y):
#     magnitud = np.sqrt(gradiente_x ** 2 + gradiente_y ** 2)
#     orientacion = np.arctan2(gradiente_y, gradiente_x) * (180 / np.pi)  # Convertir a grados
#     orientacion[orientacion < 0] += 180  # Asegurarse de que las orientaciones estén en [0, 180)
#     return magnitud, orientacion

# def Vector_histogramas(magnitud, orientacion, pixeles_por_celda=8, bins=9):
#     celdas_x = magnitud.shape[1] // pixeles_por_celda
#     celdas_y = magnitud.shape[0] // pixeles_por_celda
#     histograma_celdas = np.zeros((celdas_y, celdas_x, bins))

#     for i in range(celdas_y):
#         for j in range(celdas_x):
#             magnitud_celda = magnitud[i * pixeles_por_celda:(i + 1) * pixeles_por_celda,
#                              j * pixeles_por_celda:(j + 1) * pixeles_por_celda]
#             orientacion_celda = orientacion[i * pixeles_por_celda:(i + 1) * pixeles_por_celda,
#                                 j * pixeles_por_celda:(j + 1) * pixeles_por_celda]

#             hist, _ = np.histogram(orientacion_celda, bins=bins, range=(0, 180), weights=magnitud_celda)
#             histograma_celdas[i, j, :] = hist

#     return histograma_celdas.ravel()

# # Ejemplo de uso
# ruta_imagen = "./Personas/frameb1_0103.jpg" #es la ruta de cada uno y la imagen se se escoja
# imagen_grises = cargar_y_convertir_a_grises(ruta_imagen)
# gradiente_x, gradiente_y = calcular_gradientes(imagen_grises)
# magnitud, orientacion = calcular_magnitud_orientacion(gradiente_x, gradiente_y)
# vector_hog = Vector_histogramas(magnitud, orientacion) #Esto es la caracterización

# #detallitos (se pueden omitir en un futuro)
# print("Longitud del vector HOG:", len(vector_hog))

# # Mostrar la imagen de magnitud de gradiente como ejemplo
# plt.imshow(magnitud, cmap="gray")
# plt.title("Magnitud de Gradiente")
# plt.axis("off")
# plt.show()

