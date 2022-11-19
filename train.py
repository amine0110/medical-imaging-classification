from matplotlib import pyplot as plt
import numpy as np
import os
import cv2 
import csv
import tensorflow as tf
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
import math
from PIL import Image

from keras.optimizers import gradient_descent_v2
from keras.callbacks import EarlyStopping, ModelCheckpoint

classes_path = 'The path to the txt file for the classes'
train_dataset = 'The path to the train datasets'
input_dim = 224

def return_classes():
    with open(classes_path, 'r') as f:
        classes = f.readlines()
        classes = list(map(lambda x: x.strip(), classes))

#num_classes = len(classes)

def return_ds(input_dir):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        input_dir,               
        validation_split=0.2,        
        subset="training",           
        seed=42,                     
        image_size=(input_dim, input_dim), 
        batch_size=16, 
    label_mode='categorical',
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        input_dir,               
        validation_split=0.2,        
        subset="validation",           
        seed=42,                     
        image_size=(input_dim, input_dim), 
        batch_size=8, 
    label_mode='categorical',
    )

    return train_ds, val_ds

def return_model(input_dim, nb_classes, head=None):
    if head == 'xception' or head == 'Xception':
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(input_dim, input_dim, 3))
    
    if head == 'vgg16' or head == 'VGG16':
        base_model = VGG16(include_top=False, weights='imagenet', input_shape=(input_dim, input_dim, 3))
    
    if head == 'inceptionv3' or 'Inceptionv3' or 'InceptionV3':
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(input_dim, input_dim, 3))
    
    if head == 'densenet121':
        base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(input_dim, input_dim, 3))
    
    if not head:
        print('Please choose the pretrained model')


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='sigmoid')(x)

    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)


return_model(244, 2)