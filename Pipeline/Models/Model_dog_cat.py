# Model Architectures
import keras
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.metrics import FalsePositives as Fp, TrueNegatives as Tn, FalseNegatives as Fn, TruePositives as Tp


def seq_maxpool_cnn(log_dir):
    model_name = 'seq_maxpool_cnn'
    model = keras.Sequential([
        keras.layers.Conv2D(24, (4, 4), activation='relu', strides=1, input_shape=(150, 150, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(48, (4, 4), activation='relu', strides=1),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(96, (4, 4), activation='relu', strides=1),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='softmax')],
        name=model_name)

    plot_model(model, to_file=f'{log_dir}_{model_name}.png')

    model.compile(optimizer=Adam(lr=0.00025),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                           Fp(), Tn(), Fn(), Tp()])
    return model

# Can add multiple models
