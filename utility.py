
#import libraries
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Create the process_image function
def process_image(image_path):
    image_size = 224;
    im = Image.open(image_path)
    image = np.asarray(im)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = np.expand_dims(image.numpy(), axis=0)
    return image


def get_flower_classes(category_names,top_k_indices):
    #Read class names
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    
    #map labels with class name    
    flower_classes = []
    for idx in top_k_indices[0]:
        flower_classes.append(class_names[str(idx+1)])
    return flower_classes    