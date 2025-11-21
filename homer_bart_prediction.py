import tensorflow as tf
from tensorflow.keras.models import load_model
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "homer_bart_model.keras")
model = load_model(MODEL_PATH)

IMG_SIZE =(128,128)
def preprocess_image(bytes_data):
    #imgs = tf.io.read_file(bytes_data)
    img = tf.image.decode_image(bytes_data, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize to [0,1]
    img = tf.expand_dims(img, axis=0)  # Add batch dimension cnn need batch 
    return img    
class_name={0:'Homer',1:'Bart' }
def predict_image(model, path):
    img =preprocess_image(path)
    prob =model.predict(img)[0][0] #predict for first element that's why [0][0]
    label =1 if prob >0.5 else 0
    return class_name[label],float(prob)
