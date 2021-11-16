# Model Declaration

# from keras.models import Model
from keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def deep_diamond(log_dir):
    model_name = 'deep_diamond'
    input_layer = layers.Input(shape=(25,))  # x_train.shape[1:]
    hidden1 = layers.Dense(36, activation="elu")(input_layer)
    drop1 = layers.Dropout(rate=0.2)(hidden1)
    hidden2 = layers.Dense(18, activation="elu")(drop1)
    drop2 = layers.Dropout(rate=0.2)(hidden2)
    hidden3 = layers.Dense(8, activation="elu")(drop2)
    output = layers.Dense(1, activation="sigmoid")(hidden3)
    model = Model(inputs=[input_layer], outputs=[output], name=model_name)

    plot_model(model, to_file=f'{log_dir}__{model_name}.png')  # Save model structure image

    model.compile(optimizer=Adam(),
                  loss='mse',
                  metrics=['accuracy', 'AUC', 'Recall', 'Precision', 'mae'])
    return model


'''def deep_diamond(log_dir):
    model_name = 'deep_diamond'
    model = Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(36, activation="elu"),
        layers.Dropout(rate=0.2),
        layers.Dense(18, activation="elu"),
        layers.Dropout(rate=0.2),
        layers.Dense(8, activation="elu"),
        layers.Dense(1, activation="sigmoid")
    ])

    plot_model(model, to_file=f'{log_dir}__{model_name}.png')  # Save model structure image

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                           Fp(), Tn(), Fn(), Tp()])
    return model'''
