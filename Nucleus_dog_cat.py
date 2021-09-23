# Nucleus File

from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from Pipeline.Models import Model_dog_cat as modelDogCat
from Pipeline.Grid_Search import Grid_Search_dog_cat as gridDogCat
from Pipeline.Data_Visual import Data_Visual_dog_cat as datavizDogCat
from Pipeline.Callbacks import Callbacks_dog_cat as cbDogCat
from Pipeline.Prediction import Prediction_dog_cat as pdDogCat
from keras.models import load_model
import pandas as pd


# Model class
class CatDogModel:
    def __init__(self, version, model_name, data_dir, saved_weights_dir):
        self.data_dir = data_dir
        self.version_model_name = f'{version}_{model_name}'

        self.model_saved_weights_dir = f'{saved_weights_dir}\\{self.version_model_name}'
        self.log_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Logs\\{self.version_model_name}'
        self.metric_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Metric-Graphs\\{self.version_model_name}'

        self.train_gen = None
        self.valid_gen = None
        self.test_gen = None

        self.model = None
        self.history = None

    # Data Preprocessing
    def preprocess(self):
        self.train_gen = procDogCat.train_image_gen(self.data_dir)
        self.valid_gen = procDogCat.valid_image_gen(self.data_dir)
        self.test_gen = procDogCat.test_image_gen(self.data_dir)

    # Model Declaration
    def model_init(self):
        self.model = modelDogCat.seq_maxpool_cnn(self.log_dir)

    # Optuna Optimization
    def grid_search(self):
        gridDogCat.optuna_executor(training_data=self.train_gen, validation_data=self.valid_gen,
                                   num_epochs=55, input_shape=(150, 150, 3),
                                   save_model_dir=self.model_saved_weights_dir, log_dir=self.log_dir)

    # Training
    def training(self, callback_bool):
        if callback_bool:
            callback_list = cbDogCat.training_callbacks(self.model_saved_weights_dir, self.log_dir)
            # Custom callback cannot be appended to callback list so is simply called
            cbDogCat.model_summary_callback(self.log_dir, self.model)
        else:
            callback_list = []

        self.history = self.model.fit(self.train_gen, validation_data=self.valid_gen, batch_size=32,
                                      steps_per_epoch=20, epochs=55, callbacks=callback_list)

    # Visualization
    def graphing(self, csv_file):
        if csv_file is not None:
            metric_data = pd.read_csv(csv_file)
        else:
            metric_data = self.history

        training_data_visualization = datavizDogCat.TrainingDataVisualization(metric_data, self.metric_dir)
        training_data_visualization.loss_graph()
        training_data_visualization.error_rate_graph()
        training_data_visualization.recall_graph()
        training_data_visualization.precision_graph()
        training_data_visualization.f1_graph()
        training_data_visualization.false_positive_graph()
        training_data_visualization.false_negative_graph()
        training_data_visualization.true_positive_graph()
        training_data_visualization.true_negative_graph()
        training_data_visualization.subplot_creation(row_size=3, col_size=3)
        training_data_visualization.confusion_matrix(self.test_gen.class_indices)

    # Evaluation
    def evaluate(self, saved_weights_dir, callback_bool):
        model = load_model(saved_weights_dir)

        if callback_bool:
            callback_list = cbDogCat.evaluation_callbacks(self.log_dir)
        else:
            callback_list = []

        model.evaluate(self.test_gen,
                       batch_size=50,
                       callbacks=callback_list)

    # Evaluation Visualization
    def evaluate_graphing(self, csv_file):
        metric_data = pd.read_csv(csv_file)
        evaluation_data_visualization = datavizDogCat.EvaluationDataVisualization(metric_data, self.metric_dir)
        evaluation_data_visualization.eval_barchart()

    # Prediction
    @staticmethod
    def model_predict(saved_weights_dir, prediction_data):
        prediction = pdDogCat.PredictionDogCat(saved_weights_dir, prediction_data)
        prediction.make_prediction()


# Executor
if __name__ == '__main__':
    model_instance = CatDogModel(version='Last_Generation', model_name='dog_cat',
                                 data_dir='F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir',
                                 saved_weights_dir='F:\\Saved-Models\\Dog-Cat-Models')
    model_instance.preprocess()
    model_instance.model_init()
    # model_instance.grid_search()
    # model_instance.training(callback_bool=True)
    # model_instance.graphing(
    #     csv_file='Model-Graphs&Logs\\Model-Data_dog_cat\\Logs\\Last_Generation_dog_cat_training_metrics.csv')
    # model_instance.evaluate(saved_weights_dir='F:\\Saved-Models\\Dog-Cat-Models\\Last_Generation_dog_cat.h5',
    #                         callback_bool=True)
    # model_instance.evaluate_graphing(
    #     csv_file='Model-Graphs&Logs\\Model-Data_dog_cat\\Logs\\Last_Generation_dog_cat_evaluation_metrics.csv')
    # model_instance.model_predict(saved_weights_dir='F:\\Saved-Models\\Dog-Cat-Models\\Last_Generation_dog_cat.h5',
    #                              prediction_data='F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir\\Predict')
