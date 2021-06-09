import random
import os
import shutil
import glob


# Making directories for Dog-Cat data
def make_dir():
    os.chdir('F:\\Data-Warehouse\\Dog-Cat-Data')

    if not os.path.exists('training_dir\\Train\\Dog'):
        os.makedirs('training_dir\\Train\\Dog')

    if not os.path.exists('training_dir\\Valid\\Dog'):
        os.makedirs('training_dir\\Valid\\Dog')

    if not os.path.exists('training_dir\\Test\\Dog'):
        os.makedirs('training_dir\\Test\\Dog')

    if not os.path.exists('training_dir\\Train\\Cat'):
        os.makedirs('training_dir\\Train\\Cat')

    if not os.path.exists('training_dir\\Valid\\Cat'):
        os.makedirs('training_dir\\Valid\\Cat')

    if not os.path.exists('training_dir\\Test\\Cat'):
        os.makedirs('training_dir\\Test\\Cat')


# Moving images into respective directories
def move_images():
    os.chdir('F:\\Data-Warehouse\\Dog-Cat-Data\\Image_Vault')

    # Move 1000 random images with 'cat' in the filename
    for image in random.sample(glob.glob('cat*'), 1000):
        shutil.move(image, 'F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir\\Train\\Cat')

    for image in random.sample(glob.glob('cat*'), 500):
        shutil.move(image, 'F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir\\Valid\\Cat')

    for image in random.sample(glob.glob('cat*'), 100):
        shutil.move(image, 'F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir\\Test\\Cat')

    for image in random.sample(glob.glob('dog*'), 1000):
        shutil.move(image, 'F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir\\Train\\Dog')

    for image in random.sample(glob.glob('dog*'), 500):
        shutil.move(image, 'F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir\\Valid\\Dog')

    for image in random.sample(glob.glob('dog*'), 100):
        shutil.move(image, 'F:\\Data-Warehouse\\Dog-Cat-Data\\training_dir\\Test\\Dog')


# Executor
make_dir()
move_images()
