from keras.models import load_model
import cv2
import os
from icecream import ic


class PredictionDogCat:
    def __init__(self, saved_weights_dir, prediction_data):
        self.saved_weights_dir = saved_weights_dir
        self.prediction_data = prediction_data

    def make_prediction(self):
        ic(self.saved_weights_dir)
        ic(self.prediction_data)
        img_dir = os.listdir(self.prediction_data)
        ic(len(img_dir))

        for img in img_dir:
            test_image = cv2.imread(img)
            ic(test_image)
            ic(len(test_image))

        model = load_model(self.saved_weights_dir)
