from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
from utils import config as cfg
import numpy as np
from utils.utils import return_classes


path_to_model = 'models/my_model.h5'
img_path = 'dataset/test/dyed-lifted-polyps/0a7bdce4-ac0d-44ef-93ee-92dfc8fe0b81.jpg'

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

predic_class, predic_proba = predict_one_image(img_path, path_to_model)
print(predic_class, predic_proba)
