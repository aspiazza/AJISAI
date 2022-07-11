# Callback declaration

from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
import sys


class TrainingCallbacks:
    def __init__(self, saved_weights_dir, log_dir):
        self.saved_weights_dir = saved_weights_dir
        self.log_dir = log_dir

        self.model_checkpoint = None
        self.metric_csv = None

    def model_checkpoint_callback(self):
        self.model_checkpoint = ModelCheckpoint(
            f'{self.saved_weights_dir}.h5',
            save_best_only=True)
        return self.model_checkpoint

    def metric_csv_callback(self):
        self.metric_csv = CSVLogger(f'{self.log_dir}_training_metrics.csv', append=False, separator=',', )
        return self.metric_csv

    @staticmethod
    def reduce_lr_plateau_callback():
        reduce_plat = ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.5, min_lr=0.00001)
        return reduce_plat


def training_callbacks(saved_weights_dir, log_dir):
    callback = TrainingCallbacks(saved_weights_dir=saved_weights_dir, log_dir=log_dir)
    callback_list = [callback.model_checkpoint_callback(),
                     callback.metric_csv_callback(),
                     callback.reduce_lr_plateau_callback()]
    return callback_list


def model_summary_callback(log_dir, model):
    class ModelSummaryCallback:
        @staticmethod
        def model_summary_creation():
            with open(f'{log_dir}_summary.txt', 'w') as summary_file:
                sys.stdout = summary_file
                model.summary()
                summary_file.close()
                sys.stdout = sys.__stdout__

    ModelSummaryCallback().model_summary_creation()
