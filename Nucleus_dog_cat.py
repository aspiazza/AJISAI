from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from Pipeline.Models import Model_dog_cat as modelDogCat
from Pipeline.Data_Visual import Data_Visual_dog_cat as datavizDogCat
from keras.callbacks import CSVLogger
import keras
import tensorflow as tf
import sys
from datetime import datetime
from icecream import ic


# TODO: Clean code, add comments
# TODO: Use Numba somehow
# Model class
class CatDogModel:  # Include logging and data viz throughout
    def __init__(self, model_name, version, datafile):
        self.model_name = model_name
        self.version = version
        self.datafile = datafile
        self.log_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Logs\\{model_name}_{version}'
        self.metric_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Metric-Graphs\\{model_name}_{version}'

    def preprocess(self):
        self.train_gen = procDogCat.train_image_gen(self.datafile)
        self.valid_gen = procDogCat.valid_image_gen(self.datafile)
        self.test_gen = procDogCat.test_image_gen(self.datafile)

    def model(self):
        self.model = modelDogCat.seq_maxpool_cnn()

        self.model_checkpoint = keras.callbacks.ModelCheckpoint(
            f'F:\\Saved-Models\\{self.version}_{self.model_name}.h5', save_best_only=True)

        self.metric_csv = CSVLogger(f'{self.log_dir}_training_metrics.csv', append=True, separator=',')

    def training(self, callback_bool):
        if callback_bool:
            callback_list = [self.metric_csv]  # self.model_checkpoint
        else:
            callback_list = []

        self.history = self.model.fit(self.train_gen,
                                      validation_data=self.valid_gen,
                                      batch_size=20,
                                      steps_per_epoch=40,
                                      epochs=3,
                                      callbacks=callback_list)

    def model_summary(self):  # TODO: Make into callback
        with open(f'{self.log_dir}_summary.txt', 'a') as log_file:
            sys.stdout = log_file
            self.model.summary()
            log_file.close()

    # TODO: Subplot module and implement more metrics
    def training_graphs(self):  # TODO: Implement csv option argument
        self.data_visualization = datavizDogCat.DataVisualization(self.history, self.metric_dir)
        self.data_visualization.loss_graph()
        self.data_visualization.error_rate_graph()
        self.data_visualization.recall_graph()
        self.data_visualization.precision_graph()
        self.data_visualization.f1_graph()
        self.data_visualization.subplot_creation(context='Training', row_size=3, col_size=2)

    def predict(self):  # TODO: Prediction graphing and implement more metrics
        test_labels = self.test_gen.classes
        print(test_labels)


# Executor
if __name__ == '__main__':
    model_instance = CatDogModel(model_name="dog_cat", version="First_Generation",
                                 datafile='F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir')
    model_instance.preprocess()
    model_instance.model()
    # model_instance.model_summary()
    model_instance.training(callback_bool=True)
    model_instance.training_graphs()
    # model_instance.predict()
