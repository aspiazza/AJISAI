from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from Pipeline.Models import Model_dog_cat as modelDogCat
from Pipeline.Data_Visual import Data_Visual_dog_cat as datavizDogCat
from datetime import datetime
import keras.callbacks
import json
import sys


# Model class
class CatDogModel:  # Include logging and data viz throughout
    def __init__(self, model_name, data):
        self.datafile = data
        self.model_name = model_name
        self.log_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Logs\\Log-{model_name}_{str(time)}'

    def preprocess(self):
        self.train_gen = procDogCat.train_image_gen(self.datafile)
        self.valid_gen = procDogCat.valid_image_gen(self.datafile)

    def model(self):
        self.model = modelDogCat.seq_maxpool_cnn()
        self.callback = keras.callbacks.ModelCheckpoint(f'\\Saved-Models\\{self.model_name}.h5', save_best_only=True)

    def training(self, ):
        training_model = self.model

        self.history = training_model.fit(self.train_gen,  # TODO: Output training results to screen and file
                                          validation_data=self.valid_gen,
                                          steps_per_epoch=50,
                                          epochs=50,
                                          validation_steps=100,
                                          callbacks=[self.callback])

        with open(f'{self.log_dir}.txt', 'a') as log_file:
            sys.stdout = log_file
            training_model.summary()
            log_file.close()

    def data_visual(self):
        return None

        # TODO: Print different metrics and log
        # TODO: Plot training results
        # TODO: Create pre-filled nucleus files


# Executor
if __name__ == '__main__':
    time = datetime.now().strftime('%H-%M-%S')
    data_directory = 'F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir'
    model_instance = CatDogModel("dog_cat", data_directory)
    model_instance.preprocess()
    model_instance.model()
    model_instance.training()
