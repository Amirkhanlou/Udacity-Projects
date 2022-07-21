import tensorflow as tf
import tensorflow_hub as hub

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument ('input', action="store",default='./test_images/orange_dahlia.jpg', help = 'Image Path', type = str)
parser.add_argument('model', action="store", default='./amir_model.h5', help='Model path', type=str)
parser.add_argument ('--top_k', action="store", default = 5, help = 'Top K most likely classes.', type = int)
parser.add_argument ('--class_names' , action="store", default = './label_map.json', help = 'json file mapping of labels', type = str)

commands = parser.parse_args()

image_path = commands.input
export_path_keras = commands.model
classes = commands.class_names
top_k = commands.top_k

#Load model
reloaded_keras_model = tf.keras.models.load_model(export_path_keras, 
                                                  custom_objects={'KerasLayer':hub.KerasLayer}, compile= False)

# Label Mapping
with open(classes, 'r') as f:
    class_names = json.load(f)

    
# Create the process_image function
def process_image(img):
    img = np.squeeze(img)
    image = tf.image.resize(img, (224, 224))   
        
    #normalize
    image /= 255
    return image 

# Predict function
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    
    image = process_image(image)
    image = np.expand_dims(image,axis=0)
    
    ps = model.predict(image)
    
    values, indices  = tf.math.top_k(ps, k=top_k)
    return values.numpy().squeeze(), indices.numpy().squeeze()

print('\n Top {top_k} Classes \n')
probs, classes = predict(image_path, reloaded_keras_model, top_k)

for p, c in zip(probs, classes):
    print('Label:', c)
    print('Class name:', class_names[str(c+1)].title())
    print('Probablity:',p)