# Data Exploration

import pandas as pd

csv_directory = 'F:\\Data-Warehouse\\Diamonds-Data\\diamonds.csv'
metric_graphs_dir = 'C:\\Users\\17574\\PycharmProjects\\Kraken\\AJISAI-Project\\Model-Graphs&Logs\\Model-Data_diamond\\Metric-Graphs'

diamonds_csv = pd.read_csv(csv_directory)


def count_unique_values(csv_data):
    cut_data = csv_data["cut"].value_counts(normalize=True)
    color_data = csv_data["color"].value_counts(normalize=True)
    clarity_data = csv_data["clarity"].value_counts(normalize=True)

    return cut_data, color_data, clarity_data


cut, color, clarity = count_unique_values(diamonds_csv)

print(color)
