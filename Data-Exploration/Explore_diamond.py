# Data Exploration

import numpy as np
from pandas import read_csv
from plotly import offline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from icecream import ic

csv_directory = 'F:\\Data-Warehouse\\Diamonds-Data\\diamonds.csv'
metric_graphs_dir = 'C:\\Users\\17574\\PycharmProjects\\Kraken\\AJISAI-Project\\Model-Graphs&Logs\\Model-Data_diamond\\Metric-Graphs'

diamonds_csv = read_csv(csv_directory)


def value_count_extractor(csv_data, data_feature):
    feature = csv_data[data_feature]
    unique_feature_count = feature.value_counts(normalize=True)
    feature_percentage_count = unique_feature_count.values
    feature_data = unique_feature_count.keys()

    # If data is not a string (int or float)
    if not isinstance(feature[0], str):
        percentage = (max(feature) - min(feature)) / 5
        one_to_twenty_percent = []
        twenty_to_forty_percent = []
        forty_to_sixty_percent = []
        sixty_to_eighty_percent = []
        eighty_to_hundred_percent = []

        for data in feature:
            if data <= min(feature) + (percentage * 1):
                one_to_twenty_percent.append(data)

            elif data <= min(feature) + (percentage * 2):
                twenty_to_forty_percent.append(data)

            elif data <= min(feature) + (percentage * 3):
                forty_to_sixty_percent.append(data)

            elif data <= min(feature) + (percentage * 4):
                sixty_to_eighty_percent.append(data)

            elif data <= min(feature) + (percentage * 5):
                eighty_to_hundred_percent.append(data)

        return len(one_to_twenty_percent), len(twenty_to_forty_percent), len(forty_to_sixty_percent), \
               len(sixty_to_eighty_percent), len(eighty_to_hundred_percent)

    else:
        unique_feature_count = feature_percentage_count * len(diamonds_csv)
        return unique_feature_count, list(feature_data)


def variance_extractor(csv_data, data_feature):
    pass


def correlation_value_extractor(csv_data, data_feature):
    pass


table_one_to_twenty_percent, table_twenty_to_forty_percent, table_forty_to_sixty_percent, \
table_sixty_to_eighty_percent, table_eighty_to_hundred_percent = value_count_extractor(diamonds_csv, "table")

carat_one_to_twenty_percent, carat_twenty_to_forty_percent, carat_forty_to_sixty_percent, \
carat_sixty_to_eighty_percent, carat_eighty_to_hundred_percent = value_count_extractor(diamonds_csv, "carat")

dp_one_to_twenty_percent, dp_twenty_to_forty_percent, dp_forty_to_sixty_percent, \
dp_sixty_to_eighty_percent, dp_eighty_to_hundred_percent = value_count_extractor(diamonds_csv, "depth_percent")

length_one_to_twenty_percent, length_twenty_to_forty_percent, length_forty_to_sixty_percent, \
length_sixty_to_eighty_percent, length_eighty_to_hundred_percent = value_count_extractor(diamonds_csv, "length")

width_one_to_twenty_percent, width_twenty_to_forty_percent, width_forty_to_sixty_percent, \
width_sixty_to_eighty_percent, width_eighty_to_hundred_percent = value_count_extractor(diamonds_csv, "width")

depth_one_to_twenty_percent, depth_twenty_to_forty_percent, depth_forty_to_sixty_percent, \
depth_sixty_to_eighty_percent, depth_eighty_to_hundred_percent = value_count_extractor(diamonds_csv, "depth")

price_one_to_twenty_percent, price_twenty_to_forty_percent, price_forty_to_sixty_percent, \
price_sixty_to_eighty_percent, price_eighty_to_hundred_percent = value_count_extractor(diamonds_csv, "price")

cut_count, cut_categories = value_count_extractor(diamonds_csv, "cut")

clarity_count, clarity_categories = value_count_extractor(diamonds_csv, "clarity")

color_count, color_categories = value_count_extractor(diamonds_csv, "color")

exploration_figure = make_subplots(rows=5, cols=2, subplot_titles=(
    'Carat Distribution', 'Depth Percent Distribution', 'Table Size Distribution', 'Length mm Distribution',
    'Width mm Distribution', 'Depth mm Distribution', 'Price mm Distribution'))

exploration_figure.add_trace(go.Bar(x=['0 - 20%', '20 - 40%', '40 - 60%', '60 - 80%', '80 - 100%'],
                                    y=[table_twenty_to_forty_percent, table_forty_to_sixty_percent,
                                       table_sixty_to_eighty_percent, table_eighty_to_hundred_percent,
                                       table_eighty_to_hundred_percent]), row=2, col=1)

exploration_figure.add_trace(go.Bar(x=['0 - 20%', '20 - 40%', '40 - 60%', '60 - 80%', '80 - 100%'],
                                    y=[carat_twenty_to_forty_percent, carat_forty_to_sixty_percent,
                                       carat_sixty_to_eighty_percent, carat_eighty_to_hundred_percent,
                                       carat_eighty_to_hundred_percent]), row=1, col=1)

exploration_figure.add_trace(go.Bar(x=['0 - 20%', '20 - 40%', '40 - 60%', '60 - 80%', '80 - 100%'],
                                    y=[dp_twenty_to_forty_percent, dp_forty_to_sixty_percent,
                                       dp_sixty_to_eighty_percent, dp_eighty_to_hundred_percent,
                                       dp_eighty_to_hundred_percent]), row=1, col=2)

exploration_figure.add_trace(go.Bar(x=['0 - 20%', '20 - 40%', '40 - 60%', '60 - 80%', '80 - 100%'],
                                    y=[length_twenty_to_forty_percent, length_forty_to_sixty_percent,
                                       length_sixty_to_eighty_percent, length_eighty_to_hundred_percent,
                                       length_eighty_to_hundred_percent]), row=2, col=2)

exploration_figure.add_trace(go.Bar(x=['0 - 20%', '20 - 40%', '40 - 60%', '60 - 80%', '80 - 100%'],
                                    y=[width_twenty_to_forty_percent, width_forty_to_sixty_percent,
                                       width_sixty_to_eighty_percent, width_eighty_to_hundred_percent,
                                       width_eighty_to_hundred_percent]), row=3, col=1)

exploration_figure.add_trace(go.Bar(x=['0 - 20%', '20 - 40%', '40 - 60%', '60 - 80%', '80 - 100%'],
                                    y=[depth_twenty_to_forty_percent, depth_forty_to_sixty_percent,
                                       depth_sixty_to_eighty_percent, depth_eighty_to_hundred_percent,
                                       depth_eighty_to_hundred_percent]), row=3, col=2)

exploration_figure.add_trace(go.Bar(x=['0 - 20%', '20 - 40%', '40 - 60%', '60 - 80%', '80 - 100%'],
                                    y=[price_twenty_to_forty_percent, price_forty_to_sixty_percent,
                                       price_sixty_to_eighty_percent, price_eighty_to_hundred_percent,
                                       price_eighty_to_hundred_percent]), row=4, col=1)

exploration_figure.update_layout(height=1000, width=1300, title_text='Feature Distributions')

exploration_figure.show()
offline.plot(exploration_figure,
             filename=f'{metric_graphs_dir}\\Exploration_diamond.html',
             auto_open=False)
