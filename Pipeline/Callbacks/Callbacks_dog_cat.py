from keras.callbacks import CSVLogger
import keras
import sys
from icecream import ic


def callbacks(version_model_name, log_dir):
    callback_list = []

    def model_checkpoint_callback():
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            f'F:\\Saved-Models\\{version_model_name}.h5',
            save_best_only=True)
        return model_checkpoint

    # callback_list.append(model_checkpoint_callback())

    def metric_csv_callback():
        metric_csv = CSVLogger(f'{log_dir}_training_metrics.csv', append=True, separator=',')
        return metric_csv

    callback_list.append(metric_csv_callback())

    return callback_list


def model_summary_callback(log_dir, model):
    class ModelSummaryCallback:
        def model_summary_creation(self):
            with open(f'{log_dir}_summary.txt', 'w') as summary_file:
                ic(model)
                sys.stdout = summary_file
                model.summary()
                summary_file.close()
                sys.stdout = sys.__stdout__

    ModelSummaryCallback().model_summary_creation()
