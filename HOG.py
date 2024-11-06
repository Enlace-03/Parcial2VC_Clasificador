import cv2
import numpy as np
import matplotlib.pyplot as plt
def cargar_y_convertir_a_grises(ruta):
    imagen = cv2.imread(ruta)
    imagen_grises = 0.2989 * imagen[:, :, 2] + 0.5870 * imagen[:, :, 1] + 0.1140 * imagen[:, :, 0]
    return imagen_grises

def calcular_gradientes(imagen):
    # Definir los kernels de Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Inicializar las matrices para los gradientes
    gradiente_x = np.zeros_like(imagen)
    gradiente_y = np.zeros_like(imagen)

    # Aplicar convolución manualmente (sin bordes)
    for i in range(1, imagen.shape[0] - 1):
        for j in range(1, imagen.shape[1] - 1):
            # Extraer la región de 3x3
            region = imagen[i - 1:i + 2, j - 1:j + 2]
            # Calcular la convolución con los filtros Sobel
            gradiente_x[i, j] = np.sum(region * sobel_x)
            gradiente_y[i, j] = np.sum(region * sobel_y)

    return gradiente_x, gradiente_y

def calcular_magnitud_orientacion(gradiente_x, gradiente_y):
    magnitud = np.sqrt(gradiente_x ** 2 + gradiente_y ** 2)
    orientacion = np.arctan2(gradiente_y, gradiente_x) * (180 / np.pi)  # Convertir a grados
    orientacion[orientacion < 0] += 180  # Asegurarse de que las orientaciones estén en [0, 180)
    return magnitud, orientacion

def Vector_histogramas(magnitud, orientacion, pixeles_por_celda=8, bins=9):
    celdas_x = magnitud.shape[1] // pixeles_por_celda
    celdas_y = magnitud.shape[0] // pixeles_por_celda
    histograma_celdas = np.zeros((celdas_y, celdas_x, bins))

    for i in range(celdas_y):
        for j in range(celdas_x):
            magnitud_celda = magnitud[i * pixeles_por_celda:(i + 1) * pixeles_por_celda,
                             j * pixeles_por_celda:(j + 1) * pixeles_por_celda]
            orientacion_celda = orientacion[i * pixeles_por_celda:(i + 1) * pixeles_por_celda,
                                j * pixeles_por_celda:(j + 1) * pixeles_por_celda]

            hist, _ = np.histogram(orientacion_celda, bins=bins, range=(0, 180), weights=magnitud_celda)
            histograma_celdas[i, j, :] = hist

    return histograma_celdas.ravel()

# Ejemplo de uso
ruta_imagen = "./Personas/frameb1_0103.jpg" #es la ruta de cada uno y la imagen se se escoja
imagen_grises = cargar_y_convertir_a_grises(ruta_imagen)
gradiente_x, gradiente_y = calcular_gradientes(imagen_grises)
magnitud, orientacion = calcular_magnitud_orientacion(gradiente_x, gradiente_y)
vector_hog = Vector_histogramas(magnitud, orientacion) #Esto es la caracterización

#detallitos (se pueden omitir en un futuro)
print("Longitud del vector HOG:", len(vector_hog))

# Mostrar la imagen de magnitud de gradiente como ejemplo
plt.imshow(magnitud, cmap="gray")
plt.title("Magnitud de Gradiente")
plt.axis("off")
plt.show()

