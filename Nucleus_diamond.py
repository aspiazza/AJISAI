# Nucleus File
# TODO: Deal with correlation in data (Use Ridge or Lasso. Remove features)
# TODO: Deal with imbalances in data. Resample data? Up sample/down sample
# TODO: Stratified k-fold val

from Pipeline.Preprocess import Preprocess_diamond as procDiamond
from Pipeline.Callbacks import Callbacks_diamond as cbDiamond
from Pipeline.Models import Model_diamond as modelDiamond
from Pipeline.Grid_Search import Grid_Search_diamond as gridDiamond
from Pipeline.Data_Visual import Data_Visual_diamond as datavizDiamond
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

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor_pipeline = None

        self.model = None
        self.history = None

    # Data Preprocessing
    def preprocess(self):
        self.x_train, self.x_test, self.y_train, self.y_test = procDiamond.diamond_preprocess(data_dir=self.data_dir)
        # self.x_train, self.x_test, self.y_train, self.y_test = procDiamond.strat_diamond_preprocess(
        #   data_dir=self.data_dir)
        # self.x_train, self.x_test, self.y_train, self.y_test = procDiamond.feat_removal_diamond_preprocess(
        #     data_dir=self.data_dir)

    # Model Declaration
    def model_init(self):
        self.model = modelDiamond.deep_diamond(self.log_dir)

    # Optuna Optimization
    def grid_search(self):
        pass

    # Training
    def training(self, callback_bool):
        if callback_bool:
            callback_list = cbDiamond.training_callbacks(self.model_saved_weights_dir, self.log_dir)
            cbDiamond.model_summary_callback(self.log_dir, self.model)
        else:
            callback_list = []

        self.history = self.model.fit(x=self.x_train, y=self.y_train, validation_split=0.15,
                                      batch_size=15, epochs=100, callbacks=callback_list)

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
    model_instance = DiamondModel(version='first_gen', model_name='diamond',
                                  data_dir='D:\\Data-Warehouse\\Diamonds-Data\\diamonds.csv',
                                  saved_weights_dir='D:\\Saved-Models\\Diamond-Models')
    model_instance.preprocess()
    model_instance.model_init()
    # model_instance.grid_search()
    model_instance.training(callback_bool=True)
    # model_instance.graphing(
    #     csv_file='Model-Graphs&Logs\\Model-Data_dog_cat\\Logs\\Last_Generation_dog_cat_training_metrics.csv')
    # model_instance.evaluate(saved_weights_dir='F:\\Saved-Models\\Dog-Cat-Models\\Last_Generation_dog_cat_optuna.h5',
    #                         callback_bool=True)
    # model_instance.evaluate_graphing(
    #     csv_file='Model-Graphs&Logs\\Model-Data_dog_cat\\Logs\\Last_Generation_dog_cat_evaluation_metrics.csv')
    # model_instance.model_predict(saved_weights_dir='F:\\Saved-Models\\Dog-Cat-Models\\Last_Generation_dog_cat.h5',
    #                              prediction_data='F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir\\Predict')
