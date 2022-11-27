# Medical Imaging Classification
> Is Frozen Weight Transfer Learning Always the Answer?

This repo's main focus isn't on data processing or creating image classification models like Inception, Xception, VGG16, etc. However, it is about how you may improve your accuracy by changing just one parameter. Naturally, instructions and code for training a classification model will be given.

![image](https://user-images.githubusercontent.com/37108394/203735774-f5e0ea76-54a0-4cd9-bff4-783a6cff235e.png)


## Dataset used
The dataset used for this task was Kvasir, it is available in Kaggle, if you want to learn more about it then you can check [this link](https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset).

## Important part of the repo
In the function `return_model()`:

```Python
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
```

The pre-trained model's weights should not be frozen because doing so increases the model's capacity for learning.

## Run the inference 
I added a script [`video_classification.py`](https://github.com/amine0110/medical-imaging-classification/blob/main/video_classification.py) that helps you do the inference in a video using one of your models. And I added also a script [`predict_options.py`](https://github.com/amine0110/medical-imaging-classification/blob/main/predict_options.py) that contains multiple inference options such as:
- Predict one image
- Predict one array
- Predict a folder
- Predict a video
- Predict from the webcam

## A Beginner Guide to Medical Imaging
[Join the waitlist](https://astounding-teacher-3608.ck.page/6dff57e0b5
) for the FREE medical imaging ebook!

This ebook serves as a guide for those who are new to the profession of medical imaging. It provides definitions and resources like where to learn anatomy. Where can you find quality papers? Where can you find good, cost-free datasets? plus more.

## 🆕 NEW (coming soon) 

Full course about medical imaging segmentation is coming soon, join the waitlist here:

https://pycad.co/monai-and-pytoch-for-medical-imaging/
