# Model Architectures
import keras
from keras.optimizers import SGD


def seq_maxpool_cnn():
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')],
        name='seq_maxpool_cnn')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def small_seq_cnn():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                            input_shape=(200, 200, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(1, activation='sigmoid')],
        name='small_seq_cnn')
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
