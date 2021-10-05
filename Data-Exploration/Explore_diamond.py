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
    feature = csv_data[data_feature]  # Whole Column
    unique_feature_count = feature.value_counts(normalize=True)  # Unique features
    feature_percentage_count = unique_feature_count.values  # Percentage makeup of data
    feature_data = unique_feature_count.keys()  # Data features

    # If data is not a string (int or float)
    if not isinstance(feature_data[0], str):
        percentage = (max(feature_data) - min(feature_data)) / 5

        # Create list of X labels
        x_label_range_list = []
        start = min(feature_data)

        for num in range(1, 6):
            end = min(feature_data) + (percentage * num)

            x_label_range_list.append(
                f"{('{:.4f}'.format(start))} - {('{:.4f}'.format(end))}")

            if num == 1:
                start = min(feature_data) + (percentage * num)
            else:
                start = end
            end += percentage * (num + 1)

        one_to_twenty_percent = 0
        twenty_to_forty_percent = 0
        forty_to_sixty_percent = 0
        sixty_to_eighty_percent = 0
        eighty_to_hundred_percent = 0

        for data, makeup in zip(feature_data, feature_percentage_count):
            if data <= min(feature_data) + (percentage * 1):
                one_to_twenty_percent += makeup

            elif data <= min(feature_data) + (percentage * 2):
                twenty_to_forty_percent += makeup

            elif data <= min(feature_data) + (percentage * 3):
                forty_to_sixty_percent += makeup

            elif data <= min(feature_data) + (percentage * 4):
                sixty_to_eighty_percent += makeup

            elif data <= min(feature_data) + (percentage * 5):
                eighty_to_hundred_percent += makeup

        aggregate_list = [one_to_twenty_percent, twenty_to_forty_percent, forty_to_sixty_percent,
                          sixty_to_eighty_percent, eighty_to_hundred_percent]

        return aggregate_list, x_label_range_list

    else:
        unique_feature_count = feature_percentage_count * len(diamonds_csv)
        return unique_feature_count, list(feature_data)


def variance_extractor(csv_data, data_feature):
    pass


def correlation_value_extractor(csv_data, data_feature):
    pass


carat_y_values, carat_range_list = value_count_extractor(diamonds_csv, "carat")

dp_y_values, dp_range_list = value_count_extractor(diamonds_csv, "depth_percent")

table_y_values, table_range_list = value_count_extractor(diamonds_csv, "table")

length_y_values, length_range_list = value_count_extractor(diamonds_csv, "length")

width_y_values, width_range_list = value_count_extractor(diamonds_csv, "width")

depth_y_values, depth_range_list = value_count_extractor(diamonds_csv, "depth")

price_y_values, price_range_list = value_count_extractor(diamonds_csv, "price")

cut_count, cut_categories = value_count_extractor(diamonds_csv, "cut")

clarity_count, clarity_categories = value_count_extractor(diamonds_csv, "clarity")

color_count, color_categories = value_count_extractor(diamonds_csv, "color")

exploration_figure = make_subplots(rows=5, cols=2, subplot_titles=(
    'Carat Distribution', 'Depth Percent Distribution', 'Table Size Distribution', 'Length mm Distribution',
    'Width mm Distribution', 'Depth mm Distribution', 'Price mm Distribution'))

exploration_figure.add_trace(go.Bar(x=carat_range_list,
                                    y=carat_y_values, name="Carat"), row=1, col=1)

exploration_figure.add_trace(go.Bar(x=dp_range_list,
                                    y=dp_y_values, name="Depth Percent"), row=1, col=2)

exploration_figure.add_trace(go.Bar(x=table_range_list,
                                    y=table_y_values, name="Table Size"), row=2, col=1)

exploration_figure.add_trace(go.Bar(x=length_range_list,
                                    y=length_y_values, name="Length MM"), row=2, col=2)

exploration_figure.add_trace(go.Bar(x=width_range_list,
                                    y=width_y_values, name="Width MM"), row=3, col=1)

exploration_figure.add_trace(go.Bar(x=depth_range_list,
                                    y=depth_y_values, name="Depth MM"), row=3, col=2)

exploration_figure.add_trace(go.Bar(x=price_range_list,
                                    y=price_y_values, name="Price"), row=4, col=1)

exploration_figure.update_layout(height=2000, width=1300, title_text='Feature Distributions')

exploration_figure.show()
offline.plot(exploration_figure,
             filename=f'{metric_graphs_dir}\\Exploration_diamond.html',
             auto_open=False)
