# Data Exploration for dog-cat model
import glob
import os
import random
import PIL.Image
import plotly
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots

image_directory = 'F:\\Data-Warehouse\\Dog-Cat-Data\\Image_Vault'
graph_storage_directory = 'C:\\Users\\17574\\PycharmProjects\\Kraken\\Kraken_Project\\AJISAI-Project\\Model-Graphs&Logs\\Model-Data_dog-cat\\Metric-Graphs\\'
os.chdir(image_directory)
random.seed(3)


# Plot the width and height of n amount of images
def width_height(animal, sample_size):
    width_list = []
    height_list = []
    animal_images = random.sample(glob.glob(animal), sample_size)  # Grab n amount of images
    for animal_image in animal_images:
        current_image = PIL.Image.open(animal_image)
        width, height = current_image.size  # Open image with PIL and append width/height to list
        width_list.append(width)
        height_list.append(height)
    return width_list, height_list


dog_width, dog_height = width_height('dog*', 200)
cat_width, cat_height = width_height('cat*', 200)


# Display two images side by side
def raw_comparison(animal):
    random_animal_image = random.sample(glob.glob(animal), 1)
    return Image.open(random_animal_image[0])


# Subplot creation
exploration_figure = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{'colspan': 2}, {}],
           [{'colspan': 2}, {}]],
    subplot_titles=('Dog Image', 'Cat Image',
                    'Dog Image Size Plot', None,
                    'Cat Image Size Plot', None))
exploration_figure.add_trace(go.Image(z=raw_comparison('dog*')), row=1, col=1)
exploration_figure.add_trace(go.Image(z=raw_comparison('cat*')), row=1, col=2)
exploration_figure.add_trace(go.Scatter(x=dog_width, y=dog_height, mode='markers', name='Dogs'), row=2, col=1)
exploration_figure.add_trace(go.Scatter(x=cat_width, y=cat_height, mode='markers', name='Cats'), row=3, col=1)
os.chdir(graph_storage_directory)
plotly.offline.plot(exploration_figure,
                    filename=f'{graph_storage_directory}Exploration_dog-cat.html',
                    auto_open=False)
