from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from Pipeline.Models import Model_dog_cat as modelDogCat
from Pipeline.Data_Visual import Data_Visual_dog_cat as datavizDogCat
from Pipeline.Callbacks import Callbacks_dog_cat as CbDogCat
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

    def training(self, callback_bool):
        if callback_bool:
            callback_list = CbDogCat.callbacks(self.version_model_name, self.log_dir)
            # Custom callback cannot be appended to callback list so is simply called
            CbDogCat.model_summary_callback(self.log_dir, self.model)
        else:
            callback_list = []

        self.history = self.model.fit(self.train_gen,
                                      validation_data=self.valid_gen,
                                      batch_size=20,
                                      steps_per_epoch=40,
                                      epochs=20,
                                      callbacks=callback_list)

    def predict(self, saved_weights):  # TODO: Prediction module
        if saved_weights is not None:
            self.model = self.model.load_weights(saved_weights)  # Directory of saved weights
        else:
            pass

        self.prediction_history = self.model.predict(self.test_gen,
                                                     batch_size=20)

    # TODO: Implement more metrics
    def graphing(self, csv_file, context):
        if csv_file is not None:  # If you want to use a CSV file to create graphs
            metric_data = pd.read_csv(csv_file)
        else:
            pass

        if context == 'Training':  # Context should be either 'Training' or 'Testing'
            metric_data = self.history
        elif context == 'Testing':  # TODO: METRICS EXTRACTED FROM PREDICTION HISTORY ARE NOT THE SAME AS TRAINING FUK
            metric_data = self.prediction_history
        else:
            print('Context for graphing function is not Training or Testing')

        self.data_visualization = datavizDogCat.DataVisualization(metric_data, self.metric_dir)
        self.data_visualization.loss_graph()
        self.data_visualization.error_rate_graph()
        self.data_visualization.recall_graph()
        self.data_visualization.precision_graph()
        self.data_visualization.f1_graph()
        self.data_visualization.subplot_creation(context=context, row_size=3, col_size=2)


# Executor
if __name__ == '__main__':
    model_instance = CatDogModel(model_name="dog_cat", version="First_Generation",
                                 datafile='F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir')
    model_instance.preprocess()
    model_instance.model()
    model_instance.training(callback_bool=True)
    model_instance.graphing(csv_file=None, context='Training')
    model_instance.predict(saved_weights=None)
    model_instance.graphing(csv_file=None, context='Testing')
    # model_instance.predict()
