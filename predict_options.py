from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2
from utils import config as cfg


class predict:
    def __init__(self, input_dim, classes_path, model=None, test_ds = None):
        self.model = model
        self.input_dim = input_dim
        self.test_ds = test_ds
        self.classes_path = classes_path
    
    def return_classes(self):
        with open(self.classes_path, 'r') as f:
            classes = f.readlines()
            classes = list(map(lambda x: x.strip(), classes))
    
        return classes
    
    def return_model(self, model_path):
        self.model = load_model(model_path)
    
    def return_test_ds(self, input_dir):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            input_dir,          
            seed=42,                     
            image_size=(self.input_dim, self.input_dim), 
            batch_size=16, 
        label_mode='categorical',
        )
        return test_ds

    
    def predict_one_image(self, img_path):
        classes = self.return_classes()
        img = image.load_img(img_path, target_size=(self.input_dim, self.input_dim))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_batch)
        predicted_class_idx = np.argmax(prediction)            
        probability = np.max(prediction)   
        predicted_class = classes[predicted_class_idx]  

        return predicted_class, probability
    
    def predict_one_array(self, array):
        classes = self.return_classes()
        img_array = image.img_to_array(array)
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_batch)
        predicted_class_idx = np.argmax(prediction)            
        probability = np.max(prediction)   
        predicted_class = classes[predicted_class_idx]  

        return predicted_class, probability

    def predict_directory(self, input_dir):
        self.test_ds = self.return_test_ds(input_dir)
        score = self.model.evaluate(self.test_ds, verbose=0)

        return f"La loss: {score[0]}\nL'accuracy: {score[1]}"  # loss, accuracy

    def prediction_video(self, video_path, show=False, output_path='output_video.mp4'):
        video = cv2.VideoCapture(video_path)
        success,image = video.read()

        frameSize = (image.shape[1], image.shape[0])
        out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)
        x,y,w,h = 0,0,650,150

        while success:
            success,image = video.read()
            resized = cv2.resize(image, (self.input_dim, self.input_dim))
            prediction = self.predict_one_array(resized)
            cv2.rectangle(image, (x,x), (x + w, y + h), (0,0,0), -1)
            cv2.putText(image, prediction[0], org=(x + int(w/10),y + int(h/1.5)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(255,255,255), thickness=7)
            if show:
                cv2.imshow('Output', image)

            out.write(image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        video.release()
        out.release()
        cv2.destroyAllWindows()

    def predict_webcam(self, show=False, output_path='output_video.mp4'):
        video = cv2.VideoCapture(0)
        success,image = video.read()

        frameSize = (image.shape[1], image.shape[0])
        out = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)
        x,y,w,h = 0,0,650,150

        while success:
            success,image = video.read()
            resized = cv2.resize(image, (self.input_dim, self.input_dim))
            prediction = self.predict_one_array(resized)
            cv2.rectangle(image, (x,x), (x + w, y + h), (0,0,0), -1)
            cv2.putText(image, prediction[0], org=(x + int(w/10),y + int(h/1.5)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(255,255,255), thickness=7)
            if show:
                cv2.imshow('Output', image)

            out.write(image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        video.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    ## Paths 
    model_path = "Le lien vers un des modèles"
    img_path = "Le lien vers l'image pour l'inference"
    dir_path = "Le lien vers un dossier"
    video_path = "Le lien vers la vidéo"

    ## Define the model (instance)
    instance = predict(cfg.input_dim, cfg.classes_path)
    instance.return_model(model_path)

    ## Prediction for one image
    # predictions = instance.predict_one_image(img_path)
    # print(predictions)

    ## Prediction for a folder (evaluation)
    # predictions = instance.predict_directory(dir_path)
    # print(predictions)

    ## Prediction for a video
    # instance.prediction_video(video_path, show=True)

    ## Prediction from the webcam
    # instance.predict_webcam(show=True)


    


