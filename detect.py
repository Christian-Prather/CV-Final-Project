import sys
import cv2
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
# import os
# import pathlib

# current_dir = os.getcwd()

def run_inference(input_image):
    # input = sys.argv[1]
    # input_image = pathlib.Path(current_dir + image)
    img = image.load_img(input_image, target_size=(500,500))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    model = models.load_model('model/frozen_models/frozen_keras.h5')

    # prediction = model.predict(img_array)
    prediction = model.predict_classes(img_array)

    if prediction == 0:
        class_name = "Scab"
    elif prediction == 1:
        class_name = "Rot"
    elif prediction == 2:
        class_name == "Healthy"
    elif prediction == 3:
        class_name = "Rust"
    
    return class_name 



# print("Classification: {}".format(class_name))
