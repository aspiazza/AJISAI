# Model Declaration

from keras import Sequential, layers
from keras.optimizers import Adam
from keras.metrics import FalsePositives as Fp, TrueNegatives as Tn, FalseNegatives as Fn, TruePositives as Tp
from keras.utils import plot_model


def seq_maxpool_cnn(log_dir):
    model_name = 'seq_maxpool_cnn'
    model = Sequential([
        layers.Conv2D(32, kernel_size=3, activation='selu', input_shape=(150, 150, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(rate=0.25),

        layers.Conv2D(64, kernel_size=3, activation='selu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(rate=0.25),

        layers.Conv2D(128, kernel_size=3, activation='selu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(rate=0.25),

        layers.Flatten(),
        layers.Dense(1024, activation='selu'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.5),
        layers.Dense(2, activation='softmax')],
        name=model_name)

    plot_model(model, to_file=f'{log_dir}_{model_name}.png')  # Save model structure image

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                           Fp(), Tn(), Fn(), Tp()])
    return model
