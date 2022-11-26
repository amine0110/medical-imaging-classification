from glob import glob
import os

def return_classes(classes_path):
    with open(classes_path, 'r') as f:
        classes = f.readlines()
        classes = list(map(lambda x: x.strip(), classes))
    
    return classes

def create_classes(path_datasets):
    paths = glob(path_datasets + '/*')
    file = open('utils/annotation.txt', 'w')
    for path in paths:
        file.write(os.path.basename(path) + '\n')
    
    file.close()

