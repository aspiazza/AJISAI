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


from keras.models import load_model
from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from keras.callbacks import CSVLogger
from icecream import ic


def metric_csv_callback():
    metric_csv = CSVLogger('evaluation_metrics.csv', append=True, separator=',')
    return metric_csv


def evaluate(saved_weights):
    model = load_model(saved_weights)  # Directory of saved weights

    test_gen = procDogCat.test_image_gen('F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir')

    evaluation_results = model.evaluate(test_gen, batch_size=20, callbacks=metric_csv_callback())


evaluate(saved_weights='F:\\Saved-Models\\a_good_model_dog_cat.h5')