import os
import pandas as pd
import numpy as np
from PIL import Image

# Directorio que contiene las imágenes
images_dir = '/home/jacobo/Projects/UNalaDePoio/data/images'
# Archivo CSV que contiene las coordenadas de los objetos
csv_file = '/home/jacobo/Projects/UNalaDePoio/models/data.csv'
# Leer el archivo CSV en un DataFrame
df = pd.read_csv(csv_file)

df['label_name'] = df['label_name'].replace({'casa ':0, 'deforestacion':1,'mineria':2})

df.to_pickle('/home/jacobo/Projects/UNalaDePoio/models/df_image.pkl')

print(df)
