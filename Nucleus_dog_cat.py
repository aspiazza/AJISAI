from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from Pipeline.Models import Model_dog_cat as modelDogCat
from Pipeline.Data_Visual import Data_Visual_dog_cat as datavizDogCat
from datetime import datetime
from keras.callbacks import CSVLogger
import keras
import sys


# Model class
class CatDogModel:  # Include logging and data viz throughout
    def __init__(self, model_name):
        self.model_name = model_name
        self.datafile = 'F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir'
        self.log_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Logs\\{model_name}_{str(current_time)}'
        self.loss_csv = CSVLogger(f'{self.log_dir}_loss.csv', append=True, separator=',')

    def preprocess(self):
        self.train_gen = procDogCat.train_image_gen(self.datafile)
        self.valid_gen = procDogCat.valid_image_gen(self.datafile)

    def model(self):
        self.model = modelDogCat.seq_maxpool_cnn()
        self.callback = keras.callbacks.ModelCheckpoint(f'F:\\Saved-Models\\{str(current_time)}_{self.model_name}.h5',
                                                        save_best_only=True)

    def training(self):
        self.history = self.model.fit(self.train_gen,  # TODO: Output training results to screen and file
                                      validation_data=self.valid_gen,
                                      steps_per_epoch=25,  # TODO: Plot and log metrics and training
                                      epochs=25,
                                      validation_steps=100,
                                      callbacks=[self.callback,
                                                 self.loss_csv])

    def metric_logs(self):
        with open(f'{self.log_dir}_summary.txt', 'a') as log_file:
            sys.stdout = log_file
            self.model.summary()
            log_file.close()

    def metric_graph(self):
        return None


# Executor
if __name__ == '__main__':
    current_time = datetime.now().strftime('%H-%M-%S')

    model_instance = CatDogModel("dog_cat")
    model_instance.preprocess()
    model_instance.model()
    model_instance.training()
    model_instance.metric_logs()
