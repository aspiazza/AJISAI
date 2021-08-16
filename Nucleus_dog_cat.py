from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from Pipeline.Models import Model_dog_cat as modelDogCat
from Pipeline.Data_Visual import Data_Visual_dog_cat as datavizDogCat
from Pipeline.Callbacks import Callbacks_dog_cat as CbDogCat
from keras.models import load_model
import pandas as pd
from icecream import ic


# TODO: Clean code, add comments
# TODO: Update README as code progresses
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
        self.model = modelDogCat.seq_maxpool_cnn(self.log_dir)

    # TODO: Research and incorporate Optuna
    def training(self, callback_bool):
        if callback_bool:
            callback_list = CbDogCat.training_callbacks(self.version_model_name, self.log_dir)
            # Custom callback cannot be appended to callback list so is simply called
            CbDogCat.model_summary_callback(self.log_dir, self.model)
        else:
            callback_list = []

        self.history = self.model.fit(self.train_gen,
                                      validation_data=self.valid_gen,
                                      batch_size=20,
                                      steps_per_epoch=40,
                                      epochs=50,
                                      callbacks=callback_list)

    def graphing(self, csv_file):
        if csv_file is not None:
            metric_data = pd.read_csv(csv_file)
        else:
            metric_data = self.history

        self.training_data_visualization = datavizDogCat.TrainingDataVisualization(metric_data, self.metric_dir)
        self.training_data_visualization.loss_graph()
        self.training_data_visualization.error_rate_graph()
        self.training_data_visualization.recall_graph()
        self.training_data_visualization.precision_graph()
        self.training_data_visualization.f1_graph()
        self.training_data_visualization.false_positive_graph()
        self.training_data_visualization.false_negative_graph()
        self.training_data_visualization.true_positive_graph()
        self.training_data_visualization.true_negative_graph()
        self.training_data_visualization.subplot_creation(row_size=3, col_size=3)
        self.training_data_visualization.confusion_matrix(self.test_gen.class_indices)

    def evaluate(self, saved_weights_dir, callback_bool):
        if saved_weights_dir is not None:
            self.model = load_model(saved_weights_dir)
        else:
            pass

        if callback_bool:
            callback_list = CbDogCat.evaluation_callbacks(self.log_dir)
        else:
            callback_list = []

        self.evaluation_history = self.model.evaluate(self.test_gen,
                                                      batch_size=20,
                                                      callbacks=callback_list)
        print(self.evaluation_history)

    def evaluate_graphing(self, csv_file):

        metric_data = pd.read_csv(csv_file)
        self.evaluation_data_visualization = datavizDogCat.EvaluationDataVisualization(metric_data, self.metric_dir)
        self.evaluation_data_visualization.eval_barchart()

    def predict(self, saved_weights_dir):
        self.model = load_model(saved_weights_dir)

        print(self.model.predict('F:\\Data-Warehouse\\Dog-Cat-Data\\Image_Vault\\cat.11425.jpg'))
        print(self.test_gen.classes)
        print(self.test_gen.class_indices)
        test_images, test_labels = next(self.test_gen)


# Executor
if __name__ == '__main__':
    model_instance = CatDogModel(model_name="dog_cat", version="First_Generation",
                                 datafile='F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir')
    model_instance.preprocess()
    # model_instance.model()
    # model_instance.training(callback_bool=True)
    # model_instance.graphing(csv_file='Model-Graphs&Logs\\Model-Data_dog_cat\\Logs\\First_Generation_dog_cat_training_metrics.csv')
    # model_instance.evaluate(saved_weights_dir='F:\\Saved-Models\\First_Generation_dog_cat.h5', callback_bool=True)
    # model_instance.evaluate_graphing(csv_file='Model-Graphs&Logs\\Model-Data_dog_cat\\Logs\\First_Generation_dog_cat_evaluation_metrics.csv')
    model_instance.predict(saved_weights_dir='F:\\Saved-Models\\First_Generation_dog_cat.h5')
