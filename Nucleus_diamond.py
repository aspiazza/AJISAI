# Nucleus File

from Pipeline.Preprocess import Preprocess_diamond as procDiamond
from Pipeline.Models import Model_diamond as modelDiamond
from Pipeline.Grid_Search import Grid_Search_diamond as gridDiamond
from Pipeline.Data_Visual import Data_Visual_diamond as datavizDiamond
from Pipeline.Callbacks import Callbacks_diamond as cbDiamond
from Pipeline.Prediction import Prediction_diamond as pdDiamond
from keras.models import load_model
import pandas as pd


# Model class
class DiamondModel:
    def __init__(self, version, model_name, data_dir, saved_weights_dir):
        self.data_dir = data_dir
        version_model_name = f'{version}_{model_name}'

        self.model_saved_weights_dir = f'{saved_weights_dir}\\{version_model_name}'
        self.log_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Logs\\{version_model_name}'
        self.metric_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Metric-Graphs\\{version_model_name}'

    # Data Preprocessing
    def preprocess(self):
        procDiamond.diamond_preprocess(data_dir=self.data_dir)
        pass

    # Model Declaration
    def model_init(self):
        pass

    # Optuna Optimization
    def grid_search(self):
        pass

    # Training
    def training(self, callback_bool):
        if callback_bool:
            pass
        else:
            callback_list = []

    # Visualization
    def graphing(self, csv_file):
        pass

    # Evaluation
    def evaluate(self, saved_weights_dir, callback_bool):
        pass

    # Evaluation Visualization
    def evaluate_graphing(self, csv_file):
        pass

    # Prediction
    @staticmethod
    def model_predict(saved_weights_dir, prediction_data):
        pass


# Executor
if __name__ == '__main__':
    model_instance = DiamondModel(version='First_Generation', model_name='diamond',
                                  data_dir='F:\\Data-Warehouse\\Diamonds-Data\\diamonds.csv',
                                  saved_weights_dir='F:\\Saved-Models\\Diamond-Models')
    model_instance.preprocess()
    # model_instance.model_init()
    # model_instance.grid_search()
    # model_instance.training(callback_bool=True)
    # model_instance.graphing(
    #     csv_file='Model-Graphs&Logs\\Model-Data_dog_cat\\Logs\\Last_Generation_dog_cat_training_metrics.csv')
    # model_instance.evaluate(saved_weights_dir='F:\\Saved-Models\\Dog-Cat-Models\\Last_Generation_dog_cat_optuna.h5',
    #                         callback_bool=True)
    # model_instance.evaluate_graphing(
    #     csv_file='Model-Graphs&Logs\\Model-Data_dog_cat\\Logs\\Last_Generation_dog_cat_evaluation_metrics.csv')
    # model_instance.model_predict(saved_weights_dir='F:\\Saved-Models\\Dog-Cat-Models\\Last_Generation_dog_cat.h5',
    #                              prediction_data='F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir\\Predict')
