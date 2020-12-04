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
import matplotlib.pyplot as plt

print(tf.__version__)
print("Make sure keras version ends in -tf", keras.__version__)

train_data_dir = pathlib.Path("/home/christian/Documents/PlantVillage-Dataset-master/raw/color")
test_data_dir = pathlib.Path("/home/christian/Documents/PlantVillage-Dataset-master/raw/testing")

train_image_count = len(list(train_data_dir.glob('*/*.JPG')))
train_image_count = train_image_count * 0.8
test_image_count = len(list(test_data_dir.glob('*/*.JPG')))

class_abbreviations = np.array([item.name[0] for item in train_data_dir.glob('*') if (item.name != "place.txt") and (item.name!= ".DS_Store")])
# class_abbreviations = ["S", "BR", "R", "H"]

BATCH_SIZE = 16
SEED = 1

IMG_HEIGHT = 500
IMG_WIDTH = 500

CLASSES = np.array([item.name for item in train_data_dir.glob('*') if (item.name != ".DS_Store")])
print("Class Names: ", CLASSES)
number_of_classes = len(CLASSES)

LOSS = "sparse_categorical_crossentropy"
OPTIMIZER = "sgd"
EPOCHS = 10
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
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation="relu", input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation="relu"))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Conv2D(16, kernel_size=3, activation="relu"))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(number_of_classes, activation="softmax"))
    return model



def plot_image(i, predictions_array, true_labels, img):
    true_label, img = int(true_labels[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.reshape(img, (IMG_HEIGHT, IMG_WIDTH, 3))
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(CLASSES[predicted_label],
                                  100*np.max(predictions_array),
                                  CLASSES[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_labels):
    true_label = int(true_labels[i])
    plt.grid(False)
    plt.xticks(range(4), class_abbreviations)
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')




def main():
    train, validation, test = load_images()
    test_steps_per_epoch = np.math.ceil(test.samples / test.batch_size)                       
    test_images = []
    test_labels = []
    for i in range(test_image_count):
        (image, label) = next(test)
        test_images.append(image)
        test_labels.append(label)
    
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
    
    probability_model = keras.models.Sequential([model, keras.layers.Softmax()])
    # predictions = probability_model.predict(test_images)
    predictions = []
    for i in range(len(test_images)):
        predictions.append(probability_model.predict(test_images[i]))

    num_rows = 4
    num_cols = 6
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i][0], test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i][0], test_labels)
    plt.tight_layout()
    plt.savefig('model/predictions' + date_time.strftime("%Y%m%d-%H%M%S.png"))


if __name__ == "__main__":
    main()



