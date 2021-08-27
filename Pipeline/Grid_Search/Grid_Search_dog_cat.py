import keras
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from keras.metrics import FalsePositives as Fp, TrueNegatives as Tn, FalseNegatives as Fn, TruePositives as Tp
import numpy as np
import optuna
from optuna.samplers import TPESampler


class Objective(object):
    def __init__(self, training_data, validation_data, num_epochs, input_shape, saved_model_dir, log_dir):
        self.training_data = training_data
        self.validation_data = validation_data
        self.num_epochs = num_epochs
        self.input_shape = input_shape
        self.saved_model_dir = saved_model_dir
        self.log_dir = log_dir

    def __call__(self, trial):
        num_filters = trial.suggest_categorical('num_filters', [16, 32, 48, 64])
        kernel_size = trial.suggest_int('kernel_size', 2, 4)
        stride_num = trial.suggest_int('strides', 1, 2)
        activations = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh', 'selu'])
        dense_nodes = trial.suggest_categorical('num_dense_nodes', [64, 128, 512, 1024])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])

        dict_params = {
            'num_filters': num_filters,
            'kernel_size': kernel_size,
            'stride_num': stride_num,
            'activations': activations,
            'dense_nodes': dense_nodes,
            'batch_size': batch_size
        }

        model_name = 'optuna_seq_maxpool_cnn'
        model = keras.Sequential([
            keras.layers.Conv2D(filters=dict_params['num_filters'],
                                kernel_size=dict_params['kernel_size'],
                                activation=dict_params['activations'],
                                strides=dict_params['stride_num'],
                                input_shape=self.input_shape),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(filters=dict_params['num_filters'],
                                kernel_size=dict_params['kernel_size'],
                                activation=dict_params['activations'],
                                strides=dict_params['stride_num']),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(filters=dict_params['num_filters'],
                                kernel_size=dict_params['kernel_size'],
                                activation=dict_params['activations'],
                                strides=dict_params['stride_num']),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(dict_params['dense_nodes'], activation=dict_params['activations']),
            keras.layers.Dense(1, activation='softmax')],
            name=model_name)

        opt = Adam(lr=0.00025)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt, metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                                              Fp(), Tn(), Fn(), Tp()])

        callbacks_list = [EarlyStopping(monitor='val_loss', patience=5),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                            patience=5,
                                            verbose=0, mode='auto', min_lr=1.0e-6),
                          ModelCheckpoint(filepath=f'{self.saved_model_dir}optuna_dog_cat_model.h5',
                                          monitor='val_loss', save_best_only=True),
                          CSVLogger(f'{self.log_dir}_optuna_training_metrics.csv', append=True, separator=',')]
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
                                sampler=TPESampler(n_startup_trials=25))

    study.optimize(objective, timeout=14400)

    df_results = study.trials_dataframe()
    df_results.to_csv(f'{log_dir}_optuna_results.csv')
