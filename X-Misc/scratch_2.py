from keras.models import load_model
from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from keras.callbacks import CSVLogger
from icecream import ic


def metric_csv_callback():
    metric_csv = CSVLogger('evaluation_metrics.csv', append=True, separator=',')
    return metric_csv


def evaluate(saved_weights):
    model = load_model(saved_weights)  # Directory of saved weights

    test_gen = procDogCat.test_image_gen('F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir')

    evaluation_results = model.evaluate(test_gen, batch_size=20, callbacks=metric_csv_callback())


evaluate(saved_weights='F:\\Saved-Models\\a_good_model_dog_cat.h5')
