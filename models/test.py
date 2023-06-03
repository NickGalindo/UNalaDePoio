import os
import pandas as pd
import numpy as np
from PIL import Image

# Directorio que contiene las imágenes
images_dir = 'UNalaDePoio\data\images'

# Archivo CSV que contiene las coordenadas de los objetos
csv_file = 'C:\Users\USUARIO\Desktop\codefest\UNalaDePoio\models\data.csv'  # Reemplaza con la ruta al archivo CSV que contiene las coordenadas

# Leer el archivo CSV en un DataFrame
df = pd.read_csv(csv_file)

# Crear listas para almacenar los datos de las imágenes y las etiquetas
image_data = []
labels = []

# Iterar sobre cada fila del DataFrame
for index, row in df.iterrows():
    # Obtener el nombre de la imagen y las coordenadas
    image_name = row['image_name']
    x = row['bbox_x']
    y = row['bbox_y']
    width = row['bbox_width']
    height = row['bbox_height']

    # Ruta completa de la imagen
    image_path = os.path.join(images_dir, image_name)

    # Abrir la imagen utilizando PIL (Python Imaging Library)
    image = Image.open(image_path)

    # Recortar la región de interés (ROI) basada en las coordenadas
    roi = image.crop((x, y, x + width, y + height))

    # Redimensionar la imagen de la región de interés al tamaño deseado (opcional)
    roi = roi.resize((224, 224))

    # Convertir la imagen de la región de interés a un arreglo NumPy
    roi_array = np.array(roi)

    # Agregar los datos de la imagen y la etiqueta a las listas
    image_data.append(roi_array)
    labels.append(row['label_name'])

# Crear un nuevo DataFrame con los datos de las imágenes y las etiquetas
data = pd.DataFrame({'image_data': image_data, 'label': labels})

# Guardar el DataFrame en un archivo CSV
data.to_csv('data.csv', index=False)  # Reemplaza 'data.csv' con el nombre deseado para el archivo CSV