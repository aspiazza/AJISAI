# Model Architectures
import keras
from keras.optimizers import SGD, RMSprop
from keras.metrics import TruePositives as Tp, FalsePositives as Fp, TrueNegatives as Tn, FalseNegatives as Fn


def seq_maxpool_cnn():
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
        name='seq_maxpool_cnn')

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                           Tp(), Fp(), Tn(), Fn()])
    return model


def small_seq_cnn():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                            input_shape=(300, 300, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(1, activation='sigmoid')],
        name='small_seq_cnn')
    opt = SGD(lr=0.001, momentum=0.9)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
