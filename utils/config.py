# Data paths
classes_path = 'utils/annotation.txt'
train_dataset_path = './dataset/images/train'
test_dataset_path = './dataset/images/test'
path_to_model = '/models/model_5.h5'


# Training configs
input_dim = 299   # 299, 224
epochs = 40
lr=0.01
head = 'xception' # xception, vgg16, inceptionv3, densenet121