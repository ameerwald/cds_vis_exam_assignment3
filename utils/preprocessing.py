import os
import pandas as pd
# tf tools
import tensorflow as tf
# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model
# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD
#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
# for plotting
import numpy as np
import matplotlib.pyplot as plt
# saving models 
from joblib import dump


# function from class to plot the history across epochs 
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.savefig('out/plot_history.png')
    plt.show()



def load_data():
    # load in the test data
    filename = os.path.join("data", "test_data.json")
    test_data = pd.read_json(filename, lines=True)
    # load the train data
    filename2 = os.path.join("data", "train_data.json")
    train_data = pd.read_json(filename2, lines=True)
    # load the validation data
    filename3 = os.path.join("data", "val_data.json")
    val_data = pd.read_json(filename3, lines=True)
    # initialize label names for the dataset
    labelNames = test_data['class_label'].unique()
    return test_data, val_data, train_data, labelNames

# load the model
def load_model():
    model = VGG16()
    # load model without classifier layers
    model = VGG16(include_top=False, 
                pooling='avg',
                input_shape=(32, 32, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    tf.keras.backend.clear_session()
    # add new classifier layers 
    flat1 = Flatten()(model.layers[-1].output) 
    class1 = Dense(128, activation='relu')(flat1) 
    output = Dense(15, activation='softmax')(class1) 
    # define new model
    model = Model(inputs=model.inputs, 
                outputs=output)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01, 
        decay_steps=10000,
        decay_rate=0.9)
    sgd = SGD(learning_rate=lr_schedule)
    model.compile(optimizer=sgd,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model 

# function to generate additonal data
def data_generator():
    datagen = ImageDataGenerator(horizontal_flip=True, 
                                rotation_range=20,  
                                preprocessing_function = preprocess_input)
    return datagen


# training the model 
def train_model(datagen, train_data, val_data, test_data, model):
    # Split the data into categories
    train_images = datagen.flow_from_dataframe(
        dataframe=train_data,
        directory="data/",
        x_col='image_path',
        y_col='class_label',
        target_size=(224,224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        seed=42,
    )
    val_images = datagen.flow_from_dataframe(
        dataframe=val_data,
        directory="data/",
        x_col='image_path',
        y_col='class_label',
        target_size=(224,224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        seed=42,
    )
    test_images = datagen.flow_from_dataframe(
        dataframe=test_data,
        directory="data/",
        x_col='image_path',
        y_col='class_label',
        target_size=(224,224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False,
        seed=42,
    )
    batch_size = 64
    # fitting the model 
    H = model.fit(train_images,
                batch_size=32,
                validation_data=val_images,
                steps_per_epoch=train_images.samples // batch_size,
                validation_steps=val_images.samples // batch_size,
                epochs=2,
                verbose=1)
    # getting predictions 
    predictions = model.predict(test_images, batch_size=128)
    return train_images, val_images, test_images, H, predictions