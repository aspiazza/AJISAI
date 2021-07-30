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

    # TODO: Implement more metrics
    def graphing(self, csv_file, context):
        if csv_file is not None:
            training_information = pd.read_csv(csv_file)
        else:
            training_information = self.history

        self.data_visualization = datavizDogCat.DataVisualization(training_information, self.metric_dir)
        self.data_visualization.loss_graph()
        self.data_visualization.error_rate_graph()
        self.data_visualization.recall_graph()
        self.data_visualization.precision_graph()
        self.data_visualization.f1_graph()
        self.data_visualization.subplot_creation(context=context, row_size=3, col_size=2)

    def predict(self):  # TODO: Prediction module
        self.prediction = self.model.predict(self.test_gen,
                                             batch_size=20)  # TODO: Add ability to use saved model weights
        ic(self.model.predict_classes(self.test_gen))
        ic(self.model.predict_proba(self.test_gen))
        ic(self.prediction)
        test_labels = self.test_gen.classes
        ic(test_labels)


# Executor
if __name__ == '__main__':
    model_instance = CatDogModel(model_name="dog_cat", version="First_Generation",
                                 datafile='F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir')
    model_instance.preprocess()
    model_instance.model()
    model_instance.training(callback_bool=False)
    # model_instance.graphing(csv_file=None, context='Training')
    model_instance.predict()
    # model_instance.graphing(csv_file=None, context='Testing')
    # model_instance.predict()
