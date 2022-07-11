# Data Exploration

import glob
import os
import random
import PIL.Image
from plotly import offline
import plotly.graph_objects as go
from plotly.subplots import make_subplots

image_directory = 'F:\\Data-Warehouse\\Dog-Cat-Data\\Image_Vault'
metric_graphs_dir = 'C:\\Users\\17574\\PycharmProjects\\Kraken\\AJISAI-Project\\Model-Graphs&Logs\\Model-Data_dog_cat\\Metric-Graphs'
os.chdir(image_directory)  # Change to image dir
random.seed(3)  # Select random seed for consistent results


# Plot the width and height of n amount of images
def width_height_extractor(animal, sample_size):
    image_width_list = []
    image_height_list = []
    animal_images = random.sample(glob.glob(animal), sample_size)  # Grab n amount of images
    for animal_image in animal_images:
        current_image = PIL.Image.open(animal_image)
        width_num, height_num = current_image.size  # Open image with PIL and append width/height to list
        image_width_list.append(width_num)
        image_height_list.append(height_num)
    return image_width_list, image_height_list


dog_image_width, dog_image_height = width_height_extractor('dog*', 200)
cat_image_width, cat_image_height = width_height_extractor('cat*', 200)

average_image_width = int(sum(dog_image_width + cat_image_width) / len(dog_image_width + cat_image_width))
average_image_height = int(sum(dog_image_height + cat_image_height) / len(dog_image_height + cat_image_height))


# Display two images side by side
def raw_image_comparison(animal):
    random_animal_image = random.sample(glob.glob(animal), 1)
    return PIL.Image.open(random_animal_image[0])


# Subplot creation
exploration_figure = make_subplots(rows=3, cols=2,
                                   specs=[[{}, {}],
                                          [{'colspan': 2}, {}],
                                          [{'colspan': 2}, {}]],
                                   subplot_titles=('Dog Image', 'Cat Image',
                                                   'Dog Image Size Plot', None,
                                                   'Cat Image Size Plot', None))

exploration_figure.add_trace(go.Image(z=raw_image_comparison('dog*')), row=1, col=1)
exploration_figure.add_trace(go.Image(z=raw_image_comparison('cat*')), row=1, col=2)
exploration_figure.add_trace(go.Scatter(x=dog_image_width, y=dog_image_height, mode='markers', name='Dogs'),
                             row=2, col=1)
exploration_figure.add_trace(go.Scatter(x=cat_image_width, y=cat_image_height, mode='markers', name='Cats'),
                             row=3, col=1)

exploration_figure.update_annotations(
    text=f'Average Image Size: <br> {average_image_width}x{average_image_height}',
    align='left',
    showarrow=False,
    xref='paper',
    yref='paper',
    x=0.5,
    y=0.8,
    bordercolor='grey',
    borderwidth=1)

offline.plot(exploration_figure,
             filename=f'{metric_graphs_dir}\\Exploration_dog-cat.html',
             auto_open=False)
