import cv2
import os

def procesar_video(frames_dir, output_video, frame_width=320, frame_height=180, fps=30, num_frames=605):
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
        frame_filename = f"frame_seg_{i:04d}.jpg"  # Formato de nombre de archivo
        frame_path = os.path.join(frames_dir, frame_filename)

        if not os.path.exists(frame_path):
            print(f"Frame no encontrado: {frame_path}")
            continue

        # Leer el frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"No se pudo cargar el frame: {frame_path}")
            continue

        # Escribir el frame procesado al video
        video_writer.write(frame)

    # Liberar el escritor de video
    video_writer.release()
    print(f"Video procesado guardado en: {output_video}")


#Definir las variables de entrada para procesar el video:
frames_dir = "SegmanticMAP_IMG"  #Nombre de la carpeta en que están los frames
output_video = "video_procesado_seg.mp4"    #Nombre del video de salida que reconoce a las personas
frame_width = 224                   #Ancho de los frames
frame_height = 224                      #Largo de los frames
fps = 30                                #Fotogramas por segundo
num_frames = 605                        #Número de frames


# Procesar el video
procesar_video(frames_dir, output_video, frame_width, frame_height, fps, num_frames)

