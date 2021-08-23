from keras.models import load_model
import numpy as np
import tensorflow as tf
import os


class PredictionDogCat:

    def __init__(self, saved_weights_dir, prediction_data):
        self.saved_weights_dir = saved_weights_dir
        self.prediction_data = prediction_data

    def make_prediction(self):
        images = os.listdir(self.prediction_data)
        model = load_model(self.saved_weights_dir)

        for image in images:
            print(image)

            image = tf.keras.preprocessing.image.load_img(f'{self.prediction_data}\\{image}', target_size=(150, 150))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])  # Convert single image to a batch.

            prediction = model.predict_classes(input_arr)
            print(f'Predicted:  {prediction}\n')
