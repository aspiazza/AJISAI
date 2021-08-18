from keras.models import load_model
from cv2 import imread
import os
from icecream import ic


class PredictionDogCat:
    def __init__(self, saved_weights_dir, prediction_data):
        self.saved_weights_dir = saved_weights_dir
        self.prediction_data = prediction_data

    def make_prediction(self):
        img_dir = os.listdir(self.prediction_data)

        data = imread(self.prediction_data)
        ic(data)
        ic(len(data))

        model = load_model(self.saved_weights_dir)
