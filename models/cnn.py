import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Step 1: Prepare the dataset
df = pd.read_pickle('df_image.pkl')

# Load and preprocess the images
image_paths = df['image_name'].values
images = []
carpeta_padre = os.path.dirname(os.getcwd())
PATH_IMAGES =  os.path.join(carpeta_padre, 'data', 'images')
print(PATH_IMAGES)
for image_path in PATH_IMAGES+"/"+image_paths:
    image = load_img(image_path, target_size=(720, 1280))
    image = img_to_array(image)
    images.append(image)
X = np.array(images)

print("here")

# Extract the labels from the DataFrame
y = df['label_name'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Design and build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(720, 1280, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 3: Train the CNN model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 4: Evaluate the CNN model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Step 5: Use the CNN model for predictions
# Load and preprocess new/unseen images
# Make predictions using the trained model
