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

def calcular_fondo_promedio(imagenes):
    suma_imagenes = np.zeros_like(imagenes[0], dtype=np.float32)# Inicializar una matriz para acumular las imágenes

    # Sumar todas las imágenes en la lista
    for imagen in imagenes:
        suma_imagenes += imagen.astype(np.float32)

    fondo_promedio = suma_imagenes / len(imagenes)
    fondo_promedio = fondo_promedio.astype(np.uint8)# Convertir el fondo promedio a uint8 para que sea una imagen válida
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

#Tener en cuenta para correr el archivo

#1. Descargar la carpeta de drive y gyardarla en la carpeta principal del proyecto
#2. Guardarla como SoloFondo, en su defecto cambiar la ruta en el codigo
#3. Al correr el codigo se genera una imagen llamada fondo_promedio.jpg
#   eliminarla antes de volver a correrlo una vez genrada esta por primera vez.
