from keras.callbacks import CSVLogger
import pandas as pd
import keras
import sys
import os
from icecream import ic


def training_callbacks(version_model_name, log_dir):
    callback_list = []

    def model_checkpoint_callback():
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            f'F:\\Saved-Models\\{version_model_name}.h5',
            save_best_only=True)
        return model_checkpoint

    callback_list.append(model_checkpoint_callback())

    def metric_csv_callback():
        metric_csv = CSVLogger(f'{log_dir}_training_metrics.csv', append=True, separator=',')
        return metric_csv

    callback_list.append(metric_csv_callback())
    return callback_list


def evaluation_callbacks(log_dir):  # TODO: Find out how to create custom CSVLog callback
    callback_list = []

    class CustomCSVLogger(keras.callbacks.Callback):

        def __init__(self, filename):
            super().__init__()
            self.filename = filename

        def on_test_begin(self, logs=None):
            evaluation_csv_dir = f'{log_dir}_evaluation_metrics.csv'

            if not os.path.isfile(evaluation_csv_dir):
                open(evaluation_csv_dir, mode='w').close()

            evaluation_csv = pd.read_csv(evaluation_csv_dir)

        '''def on_test_batch_begin(self, batch, logs=None):
            pass

        def on_test_batch_end(self, batch, logs=None):
            ic(logs)

        def on_test_end(self, logs=None):
            keys = list(logs.keys())
            print("Stop testing; got log keys: {}".format(keys))'''

    callback_list.append(CustomCSVLogger())
    return callback_list


def model_summary_callback(log_dir, model):
    class ModelSummaryCallback:
        def model_summary_creation(self):
            with open(f'{log_dir}_summary.txt', 'w') as summary_file:
                sys.stdout = summary_file
                model.summary()
                summary_file.close()
                sys.stdout = sys.__stdout__

    ModelSummaryCallback().model_summary_creation()
