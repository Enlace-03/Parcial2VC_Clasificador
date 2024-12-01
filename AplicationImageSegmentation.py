#This code uses the DeepLabV3 decoder and resnet101 encoder from torchvision library to perform semantic segmentation on an input image. 
#The models have been trained on COCO dataset with total of 21 classes including background.
#Model trained on the following Classes: background, aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, dining table, dog, horse,
#motorbike, person, potted plant, sheep, sofa, train, and tv/monitor. 
#You can predict and segment images only belongs to these classes using pretrained model. 
#If you want to train the segmentation models on your own image dataset to segment the images and to produce high results 
#then follow the course Deep Learning for Semantic Segmentation with Python & Pytorch, link is given in the readme.

#The code first mounts the Google Colab drive to access the image file in Colab. 
#Then, the pre-trained DeepLabV3 model is loaded using models.segmentation.deeplabv3_resnet101 from torchvision library.
import os
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#Load the pretrained Model
from torchvision import models
SegModel=models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

#Next, the image is loaded using the Image module from PIL library and displayed using matplotlib. 
#The image is then transformed using the Compose method from torchvision.transforms library. 
#The transformed image is passed through the DeepLabV3 model, which returns the segmented image in the form of a tensor.
#The segmented image is then converted to a numpy array and the unique segments are found using np.unique method.

from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

#Load Input image from google drive in Colab
img = Image.open('AltaCalidadFrames/frameb6_0048.jpg')
#plt.imshow(img); plt.show()


#Define Transformations
import torchvision.transforms as T
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
inp = trf(img).unsqueeze(0)

#Predict the output
out = SegModel(inp)['out']

predicted = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print(np.unique(predicted))

#The decode_segmap function is defined to convert the segmented image to an RGB image. The final segmented RGB image is then displayed using matplotlib.

def decode_segmap(image, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb

rgb = decode_segmap(predicted)
# Asume que 'rgb' es el array generado por decode_segmap
# def save_frame(rgb, output_path):
#     # Convertir el array NumPy a imagen
#     img = Image.fromarray(rgb.astype('uint8'), 'RGB')
#     # Guardar la imagen en el directorio especificado
#     img.save(output_path)
#     print(f"Frame guardado en: {output_path}")
#
#
# frame_output_path = os.path.join("SegmanticMAP_IMG", f"frame_seg_1.jpg")
# save_frame(rgb, frame_output_path)

#plt.imshow(rgb); plt.show()
def Segmap_video(frames_dir, output_video_seg, SegModel, frame_width=224, frame_height=224, fps=30, num_frames=605):
    # Inicializar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_seg, fourcc, fps, (frame_width, frame_height))
    for i in range(num_frames):
        frame_filename = f"frameb6_{i:04d}.jpg"  # Formato de nombre de archivo
        frame_path = os.path.join(frames_dir, frame_filename)

        if not os.path.exists(frame_path):
            print(f"Frame no encontrado: {frame_path}")
            continue

        # Leer el frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"No se pudo cargar el frame: {frame_path}")
            continue
        img = Image.open(frame_path)
        trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0)

        #Predict the output
        out = SegModel(inp)['out']

        predicted = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

        def decode_segmap(image, nc=21):

            label_colors = np.array([(0, 0, 0),  # 0=background
                                     # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                     (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                     # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                     (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                     # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                     (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                     # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                     (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

            r = np.zeros_like(image).astype(np.uint8)
            g = np.zeros_like(image).astype(np.uint8)
            b = np.zeros_like(image).astype(np.uint8)

            for l in range(0, nc):
                idx = image == l
                r[idx] = label_colors[l, 0]
                g[idx] = label_colors[l, 1]
                b[idx] = label_colors[l, 2]

            rgb = np.stack([r, g, b], axis=2)
            return rgb

        rgb = decode_segmap(predicted)

        def save_frame(rgb, output_path):
            # Convertir el array NumPy a imagen
            img = Image.fromarray(rgb.astype('uint8'), 'RGB')
            # Guardar la imagen en el directorio especificado
            img.save(output_path)

        frame_output_path = os.path.join("SegmanticMAP_IMG", f"frame_seg_{i:04d}.jpg")
        save_frame(rgb, frame_output_path)

        # Leer el frame mapeado
        frame_out = cv2.imread(frame_output_path)


        # Escribir el frame procesado al video
        video_writer.write(frame_out)

    # Liberar el escritor de video
    video_writer.release()
    print(f"Video procesado guardado en: {output_video_seg}")

def procesar_video(frames_dir, output_video, SegModel, frame_width=320, frame_height=180, fps=30, num_frames=605):
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
        frame_filename = f"frameb6_{i:04d}.jpg"  # Formato de nombre de archivo
        frame_path = os.path.join(frames_dir, frame_filename)

        if not os.path.exists(frame_path):
            print(f"Frame no encontrado: {frame_path}")
            continue

        # Leer el frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"No se pudo cargar el frame: {frame_path}")
            continue
        img = Image.open(frame_path)
        trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(),
                 T.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0)

        #Predict the output
        out = SegModel(inp)['out']

        predicted = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        Lista=np.unique(predicted)
        # Definir los colores para cada clase (aquí asumo algunos colores para los bombillos)
        colores_bombillo = {
            0: (0, 0, 0),  # background (sin bombillo)
            1: (0, 255, 255),  # aeroplane (amarillo)
            2: (0, 0, 255),  # bicycle (rojo)
            3: (255, 0, 0),  # bird (azul)
            4: (0, 255, 0),  # boat (verde)
            5: (255, 0, 255),  # bottle (magenta)
            6: (255, 165, 0),  # bus (naranja)
            7: (255, 255, 0),  # car (cyan)
            8: (255, 192, 203),  # cat (rosado)
            9: (128, 0, 128),  # chair (morado)
            10: (139, 69, 19),  # cow (marrón)
            11: (255, 215, 0),  # dining table (amarillo dorado)
            12: (255, 105, 180),  # dog (rosa fuerte)
            13: (169, 169, 169),  # horse (gris oscuro)
            14: (255, 140, 0),  # motorbike (naranja oscuro)
            15: (0, 255, 255),  # person (amarillo)
            16: (34, 139, 34),  # potted plant (verde oliva)
            17: (255, 228, 196),  # sheep (beige)
            18: (0, 128, 128),  # sofa (verde azulado)
            19: (0, 0, 255),  # train (rojo)
            20: (255, 0, 0)  # tv/monitor (azul)
        }
        for clase in Lista:
          if clase == 15:  # Si hay una persona (clase 15)
              cv2.circle(frame, (frame_width - 30, frame_height - 30), 20, colores_bombillo[15], -1)  # Bombillito amarillo
          else:
              # Dibujar un bombillito diferente dependiendo de la clase detectada
              cv2.circle(frame, (frame_width - 30, frame_height - 30), 20, colores_bombillo.get(clase, (255, 255, 255)), -1)  # Usar blanco si no está en el diccionario

        # Escribir el frame procesado al video
        video_writer.write(frame)

    # Liberar el escritor de video
    video_writer.release()
    print(f"Video procesado guardado en: {output_video}")
#Definir las variables de entrada para procesar el video:
frames_dir = "AltaCalidadFrames"   #Nombre de la carpeta en que están los frames
output_video = "VideoAltaCalidad.mp4"    #Nombre del video de salida que reconoce a las personas
frame_width = 960                       #Ancho de los frames (Altacalidad 960)(VideoBajaCalidad 320)(Segmap 640)
frame_height = 544                 #Largo de los frames (Altacalidad 544)(VideoBajaCalidad 180)(Segmap 480)
fps = 30                                #Fotogramas por segundo
num_frames = 605                        #Número de frames

output_video_seg = "VideoMapaSegmentacion.mp4"

# Procesar el video
#procesar_video(frames_dir, output_video, SegModel, frame_width, frame_height, fps, num_frames)
Segmap_video(frames_dir, output_video_seg, SegModel, fps, num_frames)


#If you want to train the segmentation models on your own image dataset to segment the images and to produce high results 
#then follow the course Deep Learning for Semantic Segmentation with Python & Pytorch, link is given in the readme..
