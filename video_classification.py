from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
from utils import config as cfg
import numpy as np
from utils.utils import return_classes
import tensorflow as tf
import cv2


def predict_one_image(img, model_path):
    classes = return_classes(cfg.classes_path)
    model = load_model(model_path)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    predicted_class_idx = np.argmax(prediction)            
    probability = np.max(prediction)   
    predicted_class = classes[predicted_class_idx]  

    return predicted_class, probability

def prediction_video(video_path, model_path):
    video = cv2.VideoCapture(video_path)
    success,image = video.read()
    count = 0

    frameSize = (1920, 1080) # it depends to the dimensions of your frames
    out = cv2.VideoWriter('output_video_30spf_direct.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)
    x,y,w,h = 0,0,650,150

    while success:
        success,image = video.read()
        count += 1
        resized = cv2.resize(image, (cfg.input_dim, cfg.input_dim))
        prediction = predict_one_image(resized, model_path)
        cv2.rectangle(image, (x,x), (x + w, y + h), (0,0,0), -1)
        cv2.putText(image, prediction[0], org=(x + int(w/10),y + int(h/1.5)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(255,255,255), thickness=7)
        cv2.imshow('Output', image)
        out.write(image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    prediction_video(cfg.video_path, cfg.path_to_model)


