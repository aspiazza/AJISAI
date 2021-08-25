# Model Architectures
import keras
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.metrics import FalsePositives as Fp, TrueNegatives as Tn, FalseNegatives as Fn, TruePositives as Tp


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

    model.compile(optimizer=RMSprop(lr=1e-05),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC', 'Recall', 'Precision',
                           Fp(), Tn(), Fn(), Tp()])
    return model

# Can add multiple models

x = Flatten()(x)
x = Dense(dict_params['num_dense_nodes'], activation='relu')(x)
x = Dropout(dict_params['drop_out'])(x)
x = Dense(dict_params['num_dense_nodes'] // dict_params['dense_nodes_divisor'],
          activation='relu')(x)





