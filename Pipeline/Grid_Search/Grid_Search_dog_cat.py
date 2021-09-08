import csv
import os
import json
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
import optuna
from keras import layers, Sequential
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.metrics import FalsePositives as Fp, TrueNegatives as Tn, FalseNegatives as Fn, TruePositives as Tp


class Objective:
    def __init__(self, training_data, validation_data, num_epochs, input_shape, saved_model_dir, log_dir):
        self.training_data = training_data
        self.validation_data = validation_data
        self.num_epochs = num_epochs
        self.input_shape = input_shape
        self.saved_model_dir = saved_model_dir
        self.log_dir = log_dir

    def __call__(self, trial):
        # Create study parameters
        num_filters_1 = trial.suggest_categorical('num_filters', [16, 32, 48, 64, 128, 256])
        num_filters_2 = trial.suggest_categorical('num_filters', [16, 32, 48, 64, 128, 256])
        num_filters_3 = trial.suggest_categorical('num_filters', [16, 32, 48, 64, 128, 256])

        kernel_size_1 = trial.suggest_int('kernel_size', 2, 4)
        kernel_size_2 = trial.suggest_int('kernel_size', 2, 4)
        kernel_size_3 = trial.suggest_int('kernel_size', 2, 4)

        stride_num_1 = trial.suggest_int('strides', 1, 2)
        stride_num_2 = trial.suggest_int('strides', 1, 2)
        stride_num_3 = trial.suggest_int('strides', 1, 2)

        activations_1 = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh', 'selu'])
        activations_2 = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh', 'selu'])
        activations_3 = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh', 'selu'])
        activations_4 = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh', 'selu'])

        dense_nodes = trial.suggest_categorical('num_dense_nodes', [32, 64, 128, 512, 1024])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 96, 128])

        # Add study params to dictionary for organization
        dict_params = {
            'num_filters_1': num_filters_1,
            'num_filters_2': num_filters_2,
            'num_filters_3': num_filters_3,

            'kernel_size_1': kernel_size_1,
            'kernel_size_2': kernel_size_2,
            'kernel_size_3': kernel_size_3,

            'stride_num_1': stride_num_1,
            'stride_num_2': stride_num_2,
            'stride_num_3': stride_num_3,

            'activations_1': activations_1,
            'activations_2': activations_2,
            'activations_3': activations_3,
            'activations_4': activations_4,

            'dense_nodes': dense_nodes,
            'batch_size': batch_size,
        }

        model_name = 'optuna_seq_maxpool_cnn'
        model = Sequential([
            layers.Conv2D(filters=dict_params['num_filters_1'],
                          kernel_size=dict_params['kernel_size_1'],
                          activation=dict_params['activations_1'],
                          strides=dict_params['stride_num_1'],
                          input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(filters=dict_params['num_filters_2'],
                          kernel_size=dict_params['kernel_size_2'],
                          activation=dict_params['activations_2'],
                          strides=dict_params['stride_num_2']),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Conv2D(filters=dict_params['num_filters_3'],
                          kernel_size=dict_params['kernel_size_3'],
                          activation=dict_params['activations_3'],
                          strides=dict_params['stride_num_3']),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),

            layers.Flatten(),
            layers.Dense(dict_params['dense_nodes'], activation=dict_params['activations_4']),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.5),
            layers.Dense(2, activation='softmax')],
            name=model_name)

        opt = Adam(lr=0.01)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt, metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                                              Fp(), Tn(), Fn(), Tp()])

        callbacks_list = [ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                            patience=5,
                                            verbose=0, mode='auto', min_lr=1.0e-6),
                          ModelCheckpoint(filepath=f'{self.saved_model_dir}_optuna.h5',
                                          monitor='val_loss', save_best_only=True),
                          CSVLogger(f'{self.log_dir}_optuna_CSVLog_training_metrics.csv', append=True, separator=',')]

        history = model.fit(x=self.training_data,
                            batch_size=dict_params['batch_size'],
                            epochs=self.num_epochs,
                            validation_data=self.validation_data,
                            callbacks=[callbacks_list])

        validation_loss = np.min(history.history['val_loss'])

        return validation_loss


def optuna_executor(training_data, validation_data, num_epochs, input_shape, save_model_dir, log_dir):
    objective = Objective(training_data=training_data, validation_data=validation_data, num_epochs=num_epochs,
                          input_shape=input_shape, saved_model_dir=save_model_dir, log_dir=log_dir)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(n_startup_trials=25))
    study.optimize(objective, timeout=14400)

    def dict_to_json(log_directory, dictionary_object):
        with open(f'{log_directory}_best_params.json', "w") as outfile:
            json.dump(dictionary_object, outfile)

    dict_to_json(log_directory=log_dir, dictionary_object=study.best_params)

    def dataframe_to_csv(log_directory, dataframe):
        dataframe.to_csv(f'{log_directory}_optuna_results.csv')  # Optuna study results

    dataframe_to_csv(log_directory=log_dir, dataframe=study.trials_dataframe())

    # Get's rid of first column in optuna generated CSV
    def csv_cleaner(log_directory):
        filepath = Path(f'{log_directory}_optuna_results.csv')

        # Create temporary file
        with open(filepath, 'r', newline='') as csv_file, NamedTemporaryFile('w', newline='', dir=filepath.parent,
                                                                             delete=False) as tmp_file:
            csv_reader = csv.reader(csv_file)
            csv_writer = csv.writer(tmp_file)

            # Copy rows of data leaving out first column.
            for row in csv_reader:
                csv_writer.writerow(row[1:])

        os.replace(tmp_file.name, filepath)  # Replace original file with updated version.

    csv_cleaner(log_directory=log_dir)
