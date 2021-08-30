# Preprocessing File
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def train_image_gen(data_directory):
    train_image_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_directory = os.path.join(data_directory, 'Train')
    train_iterator = train_image_generator.flow_from_directory(
        train_directory,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['cat', 'dog'],
        shuffle=True)
    return train_iterator


def valid_image_gen(data_directory):
    valid_image_generator = ImageDataGenerator(rescale=1. / 255.0)

    valid_directory = os.path.join(data_directory, 'Valid')
    valid_iterator = valid_image_generator.flow_from_directory(
        valid_directory,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['cat', 'dog'],
        shuffle=False)
    return valid_iterator


def test_image_gen(data_directory):
    test_image_generator = ImageDataGenerator(rescale=1. / 255.0)

    test_directory = os.path.join(data_directory, 'Test')
    test_iterator = test_image_generator.flow_from_directory(
        test_directory,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['cat', 'dog'],
        shuffle=False)
    return test_iterator
