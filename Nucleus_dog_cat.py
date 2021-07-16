from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from Pipeline.Models import Model_dog_cat as modelDogCat
from Pipeline.Data_Visual import Data_Visual_dog_cat as datavizDogCat
from keras.callbacks import CSVLogger, Callback
import keras
import tensorflow as tf
import sys
import pandas as pd
from icecream import ic


# TODO: Clean code, add comments
# TODO: Use Numba somehow
# Model class
class CatDogModel:
    def __init__(self, model_name, version, datafile):
        self.datafile = datafile
        self.version_model_name = f'{version}_{model_name}'
        self.log_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Logs\\{self.version_model_name}'
        self.metric_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Metric-Graphs\\{self.version_model_name}'

    def preprocess(self):
        self.train_gen = procDogCat.train_image_gen(self.datafile)
        self.valid_gen = procDogCat.valid_image_gen(self.datafile)
        self.test_gen = procDogCat.test_image_gen(self.datafile)

    def model(self):
        self.model = modelDogCat.seq_maxpool_cnn()

        self.model_checkpoint = keras.callbacks.ModelCheckpoint(f'F:\\Saved-Models\\{self.version_model_name}.h5',
                                                                save_best_only=True)

        self.metric_csv = CSVLogger(f'{self.log_dir}_training_metrics.csv', append=True, separator=',')

        class ModelSummaryCallback(keras.callbacks.Callback):  # TODO: Turn Model summary into a callback
            def model_summary_creation(self):
                with open(f'{self.log_dir}_summary.txt', 'a') as summary_file:
                    sys.stdout = summary_file
                    self.model.summary()
                    summary_file.close()
        self.model_summary = ModelSummaryCallback().model_summary_creation()

    def training(self, callback_bool):
        if callback_bool:
            callback_list = [self.metric_csv, self.model_checkpoint, self.model_summary]
        else:
            callback_list = []

        self.history = self.model.fit(self.train_gen,
                                      validation_data=self.valid_gen,
                                      batch_size=20,
                                      steps_per_epoch=40,
                                      epochs=1,
                                      callbacks=callback_list)

    # TODO: Implement more metrics
    def training_graphs(self, csv_file):
        if csv_file is not None:
            training_information = pd.read_csv(csv_file)
        else:
            training_information = self.history

        self.data_visualization = datavizDogCat.DataVisualization(training_information, self.metric_dir, )
        self.data_visualization.loss_graph()
        self.data_visualization.error_rate_graph()
        self.data_visualization.recall_graph()
        self.data_visualization.precision_graph()
        self.data_visualization.f1_graph()
        self.data_visualization.subplot_creation(context='Training', row_size=3, col_size=2)

    def predict(self):  # TODO: Prediction graphing module and implement more metrics
        test_labels = self.test_gen.classes
        print(test_labels)


# Executor
if __name__ == '__main__':
    model_instance = CatDogModel(model_name="dog_cat", version="First_Generation",
                                 datafile='F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir')
    model_instance.preprocess()
    model_instance.model()
    model_instance.training(callback_bool=True)
    # model_instance.training_graphs(csv_file='C:\\Users\\17574\\PycharmProjects\Kraken\\AJISAI-Project\\Model-Graphs&Logs\\Model-Data_dog_cat\\Logs\\dog_cat_13-48-44_training_metrics.csv')
    # model_instance.predict()
