# Data Exploration

import pandas as pd
from icecream import ic

csv_directory = 'F:\\Data-Warehouse\\Diamonds-Data\\diamonds.csv'
metric_graphs_dir = 'C:\\Users\\17574\\PycharmProjects\\Kraken\\AJISAI-Project\\Model-Graphs&Logs\\Model-Data_diamond\\Metric-Graphs'

diamonds_csv = pd.read_csv(csv_directory)


def value_count(csv_data, data_feature):
    feature_count = csv_data[data_feature].value_counts(normalize=True)
    feature_name = feature_count.name
    feature_value = feature_count.values
    feature_label = feature_count.keys()

    if isinstance(feature_label[0], float):  # TODO: Get range for float value features
        pass


value_count(diamonds_csv, "cut")
value_count(diamonds_csv, "carat")
# value_count(diamonds_csv, "clarity")
# value_count(diamonds_csv, "color")
