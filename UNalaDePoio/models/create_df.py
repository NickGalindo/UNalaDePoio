import os
import pandas as pd
import numpy as np
from PIL import Image

# Directory that contains the images
images_dir = '/home/jacobo/Projects/UNalaDePoio/data/images'

# CSV file that contains the object coordinates
csv_file = '/home/jacobo/Projects/UNalaDePoio/models/data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Replace label names with corresponding numeric values
df['label_name'] = df['label_name'].replace({'casa ':0, 'deforestacion':1,'mineria':2})

# Save the DataFrame as a pickle file
df.to_pickle('/home/jacobo/Projects/UNalaDePoio/models/df_image.pkl')

# Print the DataFrame
print(df)
