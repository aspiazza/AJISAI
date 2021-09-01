import optuna
import keras
import numba


def objective(trial):
    x = trial.suggest_float("x", -7, 7)  # Name and range
    y = trial.suggest_float("y", -7, 7)
    return (x - 1) ** 2 + (y + 3) ** 2


study = optuna.create_study()
study.optimize(objective, n_trials=100)  # number of iterations

print(study.best_params)  # return best parameters

study.optimize(objective, n_trials=100)  # apply optimization for another 100

print(study.best_params)

# direction="minimize" for loss
study = optuna.create_study(direction="maximize")  # Direction depends on metric


# 1. Define an objective function to be maximized.
def objective(trial):
    model = Sequential()

    # 2. Suggest values of the hyperparameters using a trial object.
    model.add(
        Conv2D(filters=trial.suggest_categorical('filters', [32, 64]),
               kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
               strides=trial.suggest_categorical('strides', [1, 2]),
               activation=trial.suggest_categorical('activation', ['relu', 'linear']),
               input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(CLASSES, activation='softmax'))

    # We compile our model with a sampled learning rate.
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
    ...
    return accuracy


# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


class Objective(object):
    def __init__(self, xcalib, ycalib, dir_save,
                 max_epochs, early_stop, learn_rate_epochs,
                 input_shape, number_of_classes):
        self.xcalib = xcalib
        self.ycalib = ycalib
        self.max_epochs = max_epochs
        self.early_stop = early_stop
        self.dir_save = dir_save
        self.learn_rate_epochs = learn_rate_epochs
        self.input_shape = input_shape
        self.number_of_classes = number_of_classes

    def __call__(self, trial):
        num_cnn_blocks = trial.suggest_int('num_cnn_blocks', 2, 4)
        num_filters = trial.suggest_categorical('num_filters', [16, 32, 48, 64])
        kernel_size = trial.suggest_int('kernel_size', 2, 4)
        num_dense_nodes = trial.suggest_categorical('num_dense_nodes', [64, 128, 512, 1024])
        dense_nodes_divisor = trial.suggest_categorical('dense_nodes_divisor', [2, 4, 8])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 96, 128])
        drop_out = trial.suggest_discrete_uniform('drop_out', 0.05, 0.5, 0.05)

        dict_params = {'num_cnn_blocks': num_cnn_blocks,
                       'num_filters': num_filters,
                       'kernel_size': kernel_size,
                       'num_dense_nodes': num_dense_nodes,
                       'dense_nodes_divisor': dense_nodes_divisor,
                       'batch_size': batch_size,
                       'drop_out': drop_out}

        # start of cnn coding
        input_tensor = Input(shape=self.input_shape)

        # 1st cnn block
        x = BatchNormalization()(input_tensor)
        x = Activation('relu')(x)
        x = Conv2D(filters=dict_params['num_filters'],
                   kernel_size=dict_params['kernel_size'],
                   strides=1, padding='same')(x)
        # x = MaxPooling2D()(x)
        x = Dropout(dict_params['drop_out'])(x)

        # additional cnn blocks
        for iblock in range(dict_params['num_cnn_blocks'] - 1):
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filters=dict_params['num_filters'],
                       kernel_size=dict_params['kernel_size'],
                       strides=1, padding='same')(x)
            x = MaxPooling2D()(x)
            x = Dropout(dict_params['drop_out'])(x)

        # mlp
        x = Flatten()(x)
        x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
        x = Dropout(dict_params['drop_out'])(x)
        x = Dense(dict_params['num_dense_nodes'] // dict_params['dense_nodes_divisor'],
                  activation='relu')(x)
        output_tensor = Dense(self.number_of_classes, activation='softmax')(x)

        # instantiate and compile model
        cnn_model = Model(inputs=input_tensor, outputs=output_tensor)
        opt = Adam(lr=0.00025)  # default = 0.001
        cnn_model.compile(loss='categorical_crossentropy',
                          optimizer=opt, metrics=['accuracy'])

        # callbacks for early stopping and for learning rate reducer
        fn = self.dir_save + str(trial.number) + '_cnn.h5'
        callbacks_list = [EarlyStopping(monitor='val_loss', patience=self.early_stop),
                          ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                            patience=self.learn_rate_epochs,
                                            verbose=0, mode='auto', min_lr=1.0e-6),
                          ModelCheckpoint(filepath=fn,
                                          monitor='val_loss', save_best_only=True)]

        # fit the model
        h = cnn_model.fit(x=self.xcalib, y=self.ycalib,
                          batch_size=dict_params['batch_size'],
                          epochs=self.max_epochs,
                          validation_split=0.25,
                          shuffle=True, verbose=0,
                          callbacks=callbacks_list)

        validation_loss = np.min(h.history['val_loss'])

        return validation_loss


maximum_epochs = 1000
early_stop_epochs = 10
learning_rate_epochs = 5
optimizer_direction = 'minimize'
number_of_random_points = 25  # random searches to start opt process
maximum_time = 4 * 60 * 60  # seconds

objective = Objective(x_calib, y_calib, results_directory,
                      maximum_epochs, early_stop_epochs,
                      learning_rate_epochs, shape_of_input, num_classes)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction=optimizer_direction,
                            sampler=TPESampler(n_startup_trials=number_of_random_points))

study.optimize(objective, timeout=maximum_time)

# save results
df_results = study.trials_dataframe()
df_results.to_pickle(results_directory + 'df_optuna_results.pkl')
df_results.to_csv(results_directory + 'df_optuna_results.csv')
