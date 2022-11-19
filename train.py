import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.optimizers import gradient_descent_v2
from keras.callbacks import ModelCheckpoint
from glob import glob
import os

classes_path = ''
dataset_path = './dataset'
input_dim = 224
epochs = 50

def return_classes(classes_path):
    with open(classes_path, 'r') as f:
        classes = f.readlines()
        classes = list(map(lambda x: x.strip(), classes))
    
    return classes

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

def return_model(input_dim, nb_classes, freeze=False, head=None):
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

    if freeze:
        for layer in base_model.layers:
            layer.trainable = False


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='sigmoid')(x)

    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    return model

def create_classes(path_datasets):
    paths = glob(path_datasets + '/*')
    file = open('utils/annotation.txt', 'w')
    for path in paths:
        file.write(os.path.basename(path) + '\n')

create_classes(dataset_path)
classes = return_classes(classes_path)
train_ds, val_ds = return_ds(dataset_path)
model = return_model(input_dim, len(classes))

model.compile(loss='categorical_crossentropy', optimizer=gradient_descent_v2.SGD(learning_rate=0.01), metrics=['accuracy'])
save_weights = ModelCheckpoint(filepath='models/my_model.h5', monitor='val_accuracy', 
                                verbose=1, save_best_only=True, save_weights_only=False, mode='max')

model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[save_weights])