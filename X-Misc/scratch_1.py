import keras

import optuna

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