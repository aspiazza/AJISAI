from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from Pipeline.Models import Model_dog_cat as modelDogCat
from Pipeline.Data_Visual import Data_Visual_dog_cat as datavizDogCat
from keras.callbacks import CSVLogger
import keras
import tensorflow as tf
import sys
from datetime import datetime


# TODO: Clean code, add comments
# TODO: Use Numba somehow
# Model class
class CatDogModel:  # Include logging and data viz throughout
    def __init__(self, model_name):
        self.model_name = model_name
        self.datafile = 'F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir'
        self.log_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Logs\\{model_name}_{str(current_time)}'

    def preprocess(self):
        self.train_gen = procDogCat.train_image_gen(self.datafile)
        self.valid_gen = procDogCat.valid_image_gen(self.datafile)

    def model(self):
        self.model = modelDogCat.seq_maxpool_cnn()

        self.model_checkpoint = keras.callbacks.ModelCheckpoint(
            f'F:\\Saved-Models\\{str(current_time)}_{self.model_name}.h5', save_best_only=True)

        self.loss_csv = CSVLogger(f'{self.log_dir}_loss.csv', append=True, separator=',')

    def training(self, callback_bool):
        if callback_bool:
            callback_list = [self.loss_csv]  # self.model_checkpoint
        else:
            callback_list = []

        self.history = self.model.fit(self.train_gen,
                                      validation_data=self.valid_gen,
                                      batch_size=20,
                                      steps_per_epoch=40,
                                      epochs=30,
                                      callbacks=callback_list)

    def model_summary(self):
        with open(f'{self.log_dir}_summary.txt', 'a') as log_file:
            sys.stdout = log_file
            self.model.summary()
            log_file.close()

    # TODO: F1, mAP, Recall, Precision, Specificity, ROC, Error Rate, Kappa graphs, CM in workbench
    def metric_graph(self):
        data_visualization = datavizDogCat.DataVis(self.history, self.model_name)
        data_visualization.loss_graph()
        # data_visualization.subplot_creation()


# Executor
if __name__ == '__main__':
    current_time = datetime.now().strftime('%H-%M-%S')
    model_instance = CatDogModel("dog_cat")
    model_instance.preprocess()
    model_instance.model()
    # model_instance.model_summary()
    model_instance.training(True)
    # model_instance.metric_graph()
