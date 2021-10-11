# Data Exploration

from numpy import sqrt
import pandas as pd
import scipy.stats as ss
import plotly.graph_objects as go
from plotly.subplots import make_subplots

csv_directory = 'F:\\Data-Warehouse\\Diamonds-Data\\diamonds.csv'
metric_graphs_dir = '..\\Model-Graphs&Logs\\Model-Data_diamond\\Metric-Graphs\\Exploration_diamond.html'
diamonds_csv = pd.read_csv(csv_directory).drop(['id'], axis=1)  # Drop ID column


def feature_distribution_graph():
    def value_count_extractor(csv_data, feature_name):
        data_feature = csv_data[feature_name]
        unique_feature_count = data_feature.value_counts(normalize=True)  # Count of unique features
        feature_percentage_count = unique_feature_count.values  # Percentage makeup of each unique feature
        feature_data = unique_feature_count.keys()  # Name of data features

        # If data is not a string (int or float)
        if not isinstance(feature_data[0], str):
            data_makeup, bin_list = pd.cut(data_feature, 5, retbins=True)

            # Create list of X labels for ranges
            x_label_range_list = []
            for index in range(1, 6):
                if index == 5:
                    break
                else:
                    x_label_range_list.append(
                        f"{('{:.4f}'.format(bin_list[index]))} - {('{:.4f}'.format(bin_list[index + 1]))}")

            one_to_twenty_percent = 0
            twenty_to_forty_percent = 0
            forty_to_sixty_percent = 0
            sixty_to_eighty_percent = 0
            eighty_to_hundred_percent = 0

            for data, makeup in zip(feature_data, feature_percentage_count):
                if data <= bin_list[0]:
                    one_to_twenty_percent += makeup

                elif data <= bin_list[1]:
                    twenty_to_forty_percent += makeup

                elif data <= bin_list[2]:
                    forty_to_sixty_percent += makeup

                elif data <= bin_list[3]:
                    sixty_to_eighty_percent += makeup

                elif data <= bin_list[4]:
                    eighty_to_hundred_percent += makeup

            aggregate_list = [one_to_twenty_percent, twenty_to_forty_percent, forty_to_sixty_percent,
                              sixty_to_eighty_percent, eighty_to_hundred_percent]

            return aggregate_list, x_label_range_list

        else:
            return feature_percentage_count, list(feature_data)

    def row_column_index_creator(index_row_size, index_col_size):
        row_col_index_list = []
        index_row_size += 1
        index_col_size += 1
        [row_col_index_list.append([row, column]) for row in range(1, index_row_size) for column in
         range(1, index_col_size)]
        return row_col_index_list

    row_col_index_list = row_column_index_creator(index_row_size=5, index_col_size=2)

    feature_distribution_figure = make_subplots(rows=5, cols=2, subplot_titles=(
        'Carat Distribution', 'Cut Quality Distribution', 'Color Distribution', 'Clarity Distribution',
        'Depth Percent Distribution', 'Table Size Distribution', 'Price mm Distribution', 'Length mm Distribution',
        'Width mm Distribution', 'Depth mm Distribution'))

    for feature, row_col in zip(diamonds_csv.columns, row_col_index_list):
        y_values, range_list = value_count_extractor(diamonds_csv, feature)

        feature_distribution_figure.add_trace(go.Bar(x=range_list, y=y_values, name=feature), row=row_col[0],
                                              col=row_col[1])

    feature_distribution_figure.update_layout(height=2000, width=1300, title_text='Feature Distributions')

    return feature_distribution_figure


def numerical_correlation_map_graph():
    cleaned_diamonds_csv = diamonds_csv.drop(['color', 'cut', 'clarity'], axis=1)
    numerical_features_dataframe = cleaned_diamonds_csv.corr()
    column_headers = cleaned_diamonds_csv.columns

    numerical_correlation_heatmap_figure = go.Figure(
        go.Heatmap(z=numerical_features_dataframe, x=column_headers, y=column_headers))
    numerical_correlation_heatmap_figure.update_layout(title_text='Numerical Feature Correlation Heatmap')

    return numerical_correlation_heatmap_figure


def categorical_correlation_map_graph():
    correlation_diamonds_csv = diamonds_csv.drop(['color', 'cut', 'clarity'], axis=1)
    column_headers = correlation_diamonds_csv.columns

    def cramers_v(feature_crosstab_matrix):
        """
        calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328

        Take two features (One of them categorical) and put them in a pd.crosstab()
        function to produce a correlation coefficient
        """
        chi2 = ss.chi2_contingency(feature_crosstab_matrix)[0]
        n = feature_crosstab_matrix.sum()
        phi2 = chi2 / n
        r, k = feature_crosstab_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        correlation_value = sqrt(phi2corr / min((k_corr - 1), (r_corr - 1)))

        return correlation_value

    def cramer_correlation_extractor_iterator(categorical_features_list):
        categorical_correlation_dataframe = pd.DataFrame(columns=[categorical_features_list], index=[column_headers])
        for categorical_feature in categorical_features_list:
            for num_feature in column_headers:
                confusion_matrix = pd.crosstab(diamonds_csv[categorical_feature], diamonds_csv[num_feature])
                categorical_correlation_dataframe.loc[num_feature, categorical_feature] = cramers_v(
                    confusion_matrix.values)

        return categorical_correlation_dataframe

    categorical_features = ['cut', 'clarity', 'color']
    categorical_feature_dataframe = cramer_correlation_extractor_iterator(categorical_features)

    categorical_correlation_heatmap_figure = go.Figure(
        go.Heatmap(z=categorical_feature_dataframe, x=categorical_features, y=column_headers))
    categorical_correlation_heatmap_figure.update_layout(title_text='Categorical Feature Correlation Heatmap')

    return categorical_correlation_heatmap_figure


def figures_to_html(figs, filename):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


figure_list = [feature_distribution_graph(), numerical_correlation_map_graph(), categorical_correlation_map_graph()]
figures_to_html(figs=figure_list, filename=metric_graphs_dir)
