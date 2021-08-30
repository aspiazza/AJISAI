# Model Architectures
import keras
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.metrics import FalsePositives as Fp, TrueNegatives as Tn, FalseNegatives as Fn, TruePositives as Tp


def seq_maxpool_cnn(log_dir):
    model_name = 'seq_maxpool_cnn'
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', strides=1, input_shape=(150, 150, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(64, (3, 3), activation='relu', strides=1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(128, (3, 3), activation='relu', strides=1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(rate=0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(1, activation='sigmoid')],
        name=model_name)

    plot_model(model, to_file=f'{log_dir}_{model_name}.png')

    model.compile(optimizer=Adam(lr=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                           Fp(), Tn(), Fn(), Tp()])
    return model

# Can add multiple models
