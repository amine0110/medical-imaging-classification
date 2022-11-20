import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import gradient_descent_v2
from keras.callbacks import ModelCheckpoint
from utils import config as cfg
from utils.utils import return_classes


def return_ds(input_dir):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        input_dir,               
        validation_split=0.2,        
        subset="training",           
        seed=42,                     
        image_size=(cfg.input_dim, cfg.input_dim), 
        batch_size=16, 
    label_mode='categorical',
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        input_dir,               
        validation_split=0.2,        
        subset="validation",           
        seed=42,                     
        image_size=(cfg.input_dim, cfg.input_dim), 
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

    print('Chosen model is:', head)

    if freeze:
        for layer in base_model.layers:
            layer.trainable = False


    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dense(1024, activation='sigmoid')(x)

    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)

    return model



if __name__ == '__main__':
    classes = return_classes(cfg.classes_path)
    train_ds, val_ds = return_ds(cfg.train_dataset_path)
    model = return_model(cfg.input_dim, len(classes), head=cfg.head, freeze=False)

    model.compile(loss='categorical_crossentropy', optimizer=gradient_descent_v2.SGD(learning_rate=cfg.lr), metrics=['accuracy'])
    save_weights = ModelCheckpoint(filepath='models/xception_trained_x_model_20_epoch.h5', monitor='val_accuracy', 
                                    verbose=1, save_best_only=True, save_weights_only=False, mode='max')

    model.fit(train_ds, epochs=cfg.epochs, validation_data=val_ds, callbacks=[save_weights])