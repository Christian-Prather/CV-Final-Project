# Classification of disease for infected apples
# Authors: George Truman, Harry Dodwell, Christian Prather
#

# General system imports (inluding tensorflow)
import sys
import cv2
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image

# Loads in input image, resizes to training image size and feeds through saved model
# Returns: string of class name
def run_inference(input_image):
    img = image.load_img(input_image, target_size=(500,500))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    # Saved model built with train.py
    model = models.load_model('model/frozen_models/frozen_keras.h5')

    # Run prediction (specifically outputing class with highest confidence)
    prediction = model.predict_classes(img_array)

    # Map prediction number to class name
    if prediction == 0:
        class_name = "Scab"
    elif prediction == 1:
        class_name = "Rot"
    elif prediction == 2:
        class_name == "Healthy"
    elif prediction == 3:
        class_name = "Rust"
    
    return class_name 


# Debugging
# print("Classification: {}".format(class_name))
