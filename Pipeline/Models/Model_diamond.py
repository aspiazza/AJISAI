# Model Declaration TODO: Fix error of missing dll file

from keras.models import Model
from keras import layers
from tensorflow.keras.optimizers import Adam
from keras.metrics import FalsePositives as Fp, TrueNegatives as Tn, FalseNegatives as Fn, TruePositives as Tp
from tensorflow.keras.utils import plot_model


def deep_diamond(log_dir):
    model_name = 'deep_diamond'
    input_layer = layers.Input(shape=(8,))  # x_train.shape[1:]
    hidden1 = layers.Dense(36, activation="elu")(input_layer)
    drop1 = layers.Dropout(rate=0.2)(hidden1)
    hidden2 = layers.Dense(18, activation="elu")(drop1)
    drop2 = layers.Dropout(rate=0.2)(hidden2)
    hidden3 = layers.Dense(8, activation="elu")(drop2)
    output = layers.Dense(1, activation="sigmoid")(hidden3)
    model = Model(inputs=[input_layer], outputs=[output], name=model_name)

    plot_model(model, to_file=f'{log_dir}__{model_name}.png')  # Save model structure image

    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                           Fp(), Tn(), Fn(), Tp()])
    return model
