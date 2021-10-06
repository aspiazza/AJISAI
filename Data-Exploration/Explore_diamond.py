# Data Exploration

import numpy as np
from pandas import read_csv
from plotly import offline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from icecream import ic

csv_directory = 'F:\\Data-Warehouse\\Diamonds-Data\\diamonds.csv'
metric_graphs_dir = '..\\Model-Graphs&Logs\\Model-Data_diamond\\Metric-Graphs\\Exploration_diamond.html'
diamonds_csv = read_csv(csv_directory)

# prints correlation arrays
diamonds_csv.corr()


def feature_distribution_graph():
    def value_count_extractor(csv_data, data_feature):
        feature = csv_data[data_feature]  # Whole Column
        unique_feature_count = feature.value_counts(normalize=True)  # Unique features
        feature_percentage_count = unique_feature_count.values  # Percentage makeup of data
        feature_data = unique_feature_count.keys()  # Data features

        # If data is not a string (int or float)
        if not isinstance(feature_data[0], str):
            percentage = (max(feature_data) - min(feature_data)) / 5

            # Create list of X labels for ranges
            x_label_range_list = []
            start = min(feature_data)

            for num in range(1, 6):
                end = min(feature_data) + (percentage * num)
                x_label_range_list.append(f"{('{:.4f}'.format(start))} - {('{:.4f}'.format(end))}")
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
            return feature_percentage_count, list(feature_data)

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

    feature_distribution_figure = make_subplots(rows=5, cols=2, subplot_titles=(
        'Carat Distribution', 'Depth Percent Distribution', 'Table Size Distribution', 'Length mm Distribution',
        'Width mm Distribution', 'Depth mm Distribution', 'Price mm Distribution', 'Cut Quality Distribution',
        'Clarity Distribution', 'Color Distribution'))

    feature_distribution_figure.add_trace(go.Bar(x=carat_range_list, y=carat_y_values, name="Carat"), row=1, col=1)

    feature_distribution_figure.add_trace(go.Bar(x=dp_range_list, y=dp_y_values, name="Depth Percent"), row=1, col=2)

    feature_distribution_figure.add_trace(go.Bar(x=table_range_list, y=table_y_values, name="Table Size"), row=2, col=1)

    feature_distribution_figure.add_trace(go.Bar(x=length_range_list, y=length_y_values, name="Length MM"), row=2,
                                          col=2)

    feature_distribution_figure.add_trace(go.Bar(x=width_range_list, y=width_y_values, name="Width MM"), row=3, col=1)

    feature_distribution_figure.add_trace(go.Bar(x=depth_range_list, y=depth_y_values, name="Depth MM"), row=3, col=2)

    feature_distribution_figure.add_trace(go.Bar(x=price_range_list, y=price_y_values, name="Price"), row=4, col=1)

    feature_distribution_figure.add_trace(go.Bar(x=cut_categories, y=cut_count, name="Cut"), row=4, col=2)

    feature_distribution_figure.add_trace(go.Bar(x=clarity_categories, y=clarity_count, name="Clarity"), row=5, col=1)

    feature_distribution_figure.add_trace(go.Bar(x=color_categories, y=color_count, name="Color"), row=5, col=2)

    feature_distribution_figure.update_layout(height=2000, width=1300, title_text='Feature Distributions')

    return feature_distribution_figure


def correlation_map_graph():
    def correlation_value_extractor(csv_data, data_feature):
        pass


def figures_to_html(figs, filename):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


figure_list = [feature_distribution_graph()]
figures_to_html(figs=figure_list, filename=metric_graphs_dir)
