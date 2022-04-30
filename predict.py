# Ignore some warnings that are not relevant 
import warnings
warnings.filterwarnings('ignore')
# Import libraries
import tensorflow as tf
import tensorflow_hub as hub
import argparse
from utility import process_image, get_flower_classes


def make_prediction(image_path,  saved_model, top_k, category_names):
    
    #Load the Keras model
    model_filepath = './{}'.format(saved_model)
    model = tf.keras.models.load_model(model_filepath  ,custom_objects={'KerasLayer':hub.KerasLayer},compile=False)

    
    #load and process image
    image = process_image(image_path)
    
    #make prediction
    probs = model.predict(image)
    top_k_probs, top_k_indices = tf.nn.top_k(probs, k=top_k)
    top_k_probs, top_k_indices = top_k_probs.numpy(), top_k_indices.numpy()
    
    #get flower classes
    flower_classes = get_flower_classes(category_names,top_k_indices)
    
    #create dictionary with class name as key and probability as value
    predictions= dict(zip(flower_classes, zip(*top_k_probs)))
    return predictions 


if __name__ == '__main__':
    print('Wait while your image is being predicted')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='image path', type=str)
    parser.add_argument('saved_model', help='saved model name', type=str)
    parser.add_argument('--top_k', default=5, help='top K most likely classes', type =int)
    parser.add_argument('--category_names',default='label_map.json', help='path to a JSON file mapping labels to flower names',type=str) 
    
    
    args = parser.parse_args()
    
    print('image_path:', args.image_path)
    print('saved_model:', args.saved_model)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    

    predictions= make_prediction(args.image_path,  args.saved_model, args.top_k, args.category_names)
    print(predictions)




