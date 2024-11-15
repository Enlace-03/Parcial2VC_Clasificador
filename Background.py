import cv2, os

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

def calcular_fondo(ruta):

    # Crear el objeto de sustracción de fondo
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    frame = cargar_imagenes_carpeta(ruta)

    for i in range(len(frame)):
        foreground_mask = bg_subtractor.apply(frame[i])

    # Obtener el modelo de fondo actual
    background_model = bg_subtractor.getBackgroundImage()
    return background_model

def HOG(ruta):
    current_frame = cv2.imread(ruta)

    # Hacer y convertir la diferencia a escala de grises
    diferencia = cv2.absdiff(current_frame, background_model)
    diferencia_gris = cv2.cvtColor(diferencia, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para obtener una máscara binaria
    _, foreground_mask = cv2.threshold(diferencia_gris, 50, 255, cv2.THRESH_BINARY)
    #Pasar a color nuevamente
    foreground_mask_color = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)

    # Aplicar la máscara a la imagen original
    imagen_resultante = cv2.bitwise_and(current_frame, foreground_mask_color)

    imagen_resultante_gris = cv2.cvtColor(imagen_resultante, cv2.COLOR_BGR2GRAY)
    vector_hog = calcular_hog(imagen_resultante_gris)

    return vector_hog

#Ejemplo de uso
#Esto solo se hace una vez, es para calcular el fondo
carpeta_imagenes_fondo = "./SoloFondo" #ruta del archivo
background_model = calcular_fondo(carpeta_imagenes_fondo)

#Esto se hace para calcular el HOG de cada imagen, es decir esto se itera
ruta = "./Personas/frameb3_0217.jpg"
vector_hog = HOG(ruta)
print(len(vector_hog))
