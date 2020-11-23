import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import pathlib
import numpy as np
import datetime
print(tf.__version__)
print("Make sure keras version ends in -tf", keras.__version__)

train_data_dir = pathlib.Path("/home/christian/Documents/PlantVillage-Dataset-master/raw/color")
test_data_dir = pathlib.Path("/home/christian/Documents/PlantVillage-Dataset-master/raw/testing")

train_image_count = len(list(train_data_dir.glob('*/*.JPG')))
train_image_count = train_image_count * 0.8



BATCH_SIZE = 16
SEED = 1

IMG_HEIGHT = 500
IMG_WIDTH = 500

CLASSES = np.array([item.name for item in train_data_dir.glob('*') if (item.name != ".DS_Store")])
print("Class Names: ", CLASSES)
number_of_classes = len(CLASSES)

LOSS = "sparse_categorical_crossentropy"
OPTIMIZER = "sgd"
EPOCHS = 1
STEPS_PER_EPOCH = np.floor(train_image_count/ BATCH_SIZE)


print("Train dir: {} BATCH_SIZE: {} SEED {}".format(train_data_dir, BATCH_SIZE, SEED))



def load_images():
    image_genrator = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 20,
                                        shear_range = 20,
                                        validation_split = 0.2)

    train_image_gen = image_genrator.flow_from_directory(directory=str(train_data_dir),
                                                        batch_size = BATCH_SIZE,
                                                        subset = "training",
                                                        seed = SEED,
                                                        color_mode = "rgb",
                                                        shuffle = True,
                                                        target_size= (IMG_HEIGHT, IMG_WIDTH),
                                                        classes = list(CLASSES),
                                                        class_mode = "sparse")
    
    validation_image_gen = image_genrator.flow_from_directory(directory=str(train_data_dir),
                                                        batch_size = BATCH_SIZE,
                                                        subset = "validation",
                                                        seed = SEED,
                                                        color_mode = "rgb",
                                                        shuffle = True,
                                                        target_size= (IMG_HEIGHT, IMG_WIDTH),
                                                        classes = list(CLASSES),
                                                        class_mode = "sparse")


    test_image_generator = ImageDataGenerator(rescale = 1./255)
    test_image_gen = test_image_generator.flow_from_directory(directory=str(test_data_dir),
                                                            batch_size = 1,
                                                            color_mode = "rgb",
                                                            seed = SEED,
                                                            target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                            classes = list(CLASSES),
                                                            class_mode = "sparse")

    return train_image_gen, validation_image_gen, test_image_gen
# Save checkpoints
model_checkpoint = ModelCheckpoint('./model/frozen_models/checkpoint.h5', 
                                    monitor='loss', 
                                    verbose=1, 
                                    save_best_only=True)

callbacks_list = [model_checkpoint]

def build_model():
    model = keras.models.Sequential()
    # model.add(keras.layers.Flatten(input_shape=[IMG_HEIGHT,IMG_WIDTH, 3]))
    # model.add(keras.layers.Dense(300, activation="relu"))
    # model.add(keras.layers.Dense(100, activation="relu"))
    # model.add(keras.layers.Dense(number_of_classes, activation = "softmax"))
    
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation="relu", input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation="relu"))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(16, kernel_size=3, activation="relu"))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(number_of_classes, activation="softmax"))

    return model


def main():
    train, validation, test = load_images()
    model = build_model()
    print(model.summary())

    model.compile(loss=LOSS,
                 optimizer = OPTIMIZER,
                 metrics = ["accuracy"])

    history = model.fit(train, 
                        steps_per_epoch = STEPS_PER_EPOCH,
                        epochs = EPOCHS,
                        validation_data= validation,
                        callbacks= callbacks_list)


    date_time = datetime.datetime.now()

    model.load_weights('./model/frozen_models/checkpoint.h5')

    model.save('./model/frozen_models/frozen_keras.h5')

    tf.saved_model.save(model, './model/frozen_models')

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    # Get frozen ConcreteFunction                                                                                                                                   
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]

    # Save frozen graph from frozen ConcreteFunction to hard drive                                                                                                  
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir="./model/frozen_models",
                    name="frozen_graph.pb",
                    as_text=False)


if __name__ == "__main__":
    main()



