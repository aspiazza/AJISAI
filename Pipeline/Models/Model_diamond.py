# Model Declaration

from keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def deep_diamond(log_dir):
    model_name = 'deep_diamond'
    input_layer = layers.Input(shape=(25,))  # x_train.shape[1:]
    hidden1 = layers.Dense(36, activation="relu")(input_layer)
    drop1 = layers.Dropout(rate=0.2)(hidden1)
    hidden2 = layers.Dense(18, activation="relu")(drop1)
    drop2 = layers.Dropout(rate=0.2)(hidden2)
    hidden3 = layers.Dense(8, activation="relu")(drop2)
    output = layers.Dense(1, activation="linear")(hidden3)
    model = Model(inputs=[input_layer], outputs=[output], name=model_name)

    plot_model(model, to_file=f'{log_dir}_{model_name}.png')  # Save model structure image

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss='mae',
                  metrics=['mae'])
    return model
