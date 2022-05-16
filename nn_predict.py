

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from convert import find_square

# Get an image
image = cv2.imread('7.png')

# Process image
target_image = find_square(image)
if target_image is None:
    print("Error did not find screen square")
    exit()
gray = tf.image.rgb_to_grayscale(target_image)
target_image = tf.image.resize(gray, [102, 85])
img_array = keras.preprocessing.image.img_to_array(target_image)
img_batch = np.expand_dims(img_array, axis=0)
#img_preprocessed = preprocess_input(img_batch)
print("\ntarget shape: %s" % str(img_batch.shape))

# Load trained model
model = keras.models.load_model("trained_model.savemodel")

# Predict image
prediction  = model.predict(img_batch)
print(prediction)
