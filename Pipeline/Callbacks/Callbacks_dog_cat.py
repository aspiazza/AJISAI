from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import sys
from icecream import ic


def training_callbacks(version_model_name, log_dir):
    callback_list = []

    def model_checkpoint_callback():
        model_checkpoint = ModelCheckpoint(
            f'F:\\Saved-Models\\{version_model_name}.h5',
            save_best_only=True)
        return model_checkpoint

    callback_list.append(model_checkpoint_callback())

    def metric_csv_callback():
        metric_csv = CSVLogger(f'{log_dir}_training_metrics.csv', append=False, separator=',', )
        return metric_csv

    callback_list.append(metric_csv_callback())

    def reduce_lr_plateau_callback():
        reduce_plat = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
        return reduce_plat

    callback_list.append(reduce_lr_plateau_callback())

    def early_stopping_callback():
        early_stop = EarlyStopping(patience=10)
        return early_stop

    callback_list.append(early_stopping_callback())

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    # lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # callback_list.append(lr_callback)

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


def evaluation_callbacks(log_dir):
    callback_list = []

    def metric_csv_callback():
        CSVLogger.on_test_begin = CSVLogger.on_train_begin  # Required for model.evaluate as CSVLogger does not work
        CSVLogger.on_test_batch_end = CSVLogger.on_epoch_end
        CSVLogger.on_test_end = CSVLogger.on_train_end

        metric_csv = CSVLogger(f'{log_dir}_evaluation_metrics.csv', append=True, separator=',')
        return metric_csv

    callback_list.append(metric_csv_callback())

    return callback_list
