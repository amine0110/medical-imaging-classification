from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
from utils import config as cfg
import numpy as np
from utils.utils import return_classes
import tensorflow as tf


def return_test_ds(input_dir):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        input_dir,          
        seed=42,                     
        image_size=(cfg.input_dim, cfg.input_dim), 
        batch_size=16, 
    label_mode='categorical',
    )
    return test_ds

def predict_one_image(img_path, model_path):
    classes = return_classes(cfg.classes_path)
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(cfg.input_dim, cfg.input_dim))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    predicted_class_idx = np.argmax(prediction)            
    probability = np.max(prediction)   
    predicted_class = classes[predicted_class_idx]  

    return predicted_class, probability

def evaluate_model(model_path):
    test_ds = return_test_ds(cfg.test_dataset_path)
    model = load_model(model_path)
    score = model.evaluate(test_ds, verbose=0)

    return score[0], score[1]  # loss, accuracy


if __name__ == '__main__':
    path_to_model = 'C:/Users/amine/Documents/Amine_Files/Hands_on_AI/Défi_1/to_send/model_6.h5'
    img_path = 'dataset/test/dyed-lifted-polyps/0a7bdce4-ac0d-44ef-93ee-92dfc8fe0b81.jpg'

    # predic_class, predic_proba = predict_one_image(img_path, path_to_model)
    # print(predic_class, predic_proba)

    loss, accuracy = evaluate_model(path_to_model)
    print('loss: ', loss)
    print('accuracy: ', accuracy)