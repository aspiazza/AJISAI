# Model Architectures
import keras
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.metrics import FalsePositives as Fp, TrueNegatives as Tn, FalseNegatives as Fn, TruePositives as Tp
import tensorflow_addons as tfa


def seq_maxpool_cnn(log_dir):
    model_name = 'seq_maxpool_cnn'
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')],
        name=model_name)

    plot_model(model, to_file=f'{log_dir}_{model_name}.png')

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                           Fp(), Tn(), Fn(), Tp()])  # TODO: Test metric tfa.metrics.RSquare()
    return model

# Can add multiple models
