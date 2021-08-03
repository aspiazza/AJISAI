from keras.models import load_model
from Pipeline.Preprocess import Preprocess_dog_cat as procDogCat
from icecream import ic


def evaluate(saved_weights):  # TODO: Testing functions
    model = load_model(saved_weights)  # Directory of saved weights

    test_gen = procDogCat.test_image_gen('F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir')

    evaluation_results = model.evaluate(test_gen, batch_size=20)

    ic(evaluation_results)

    # ic(evaluation_results['accuracy'])
    # ic(evaluation_results['loss'])
    # ic(evaluation_results['recall'])
    # ic(evaluation_results['precision'])
    # ic(evaluation_results['true_positives'])
    # ic(evaluation_results['true_negatives'])
    # ic(evaluation_results['false_positives'])
    # ic(evaluation_results['false_negatives'])
    # ic(evaluation_results['auc'])


evaluate(saved_weights='F:\\Saved-Models\\a_good_model_dog_cat.h5')
