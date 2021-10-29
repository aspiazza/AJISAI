# Data Exploration

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as ss
from sklearn.linear_model import LinearRegression

csv_directory = 'F:\\Data-Warehouse\\Diamonds-Data\\diamonds.csv'
metric_graphs_dir = '..\\Model-Graphs&Logs\\Model-Data_diamond\\Metric-Graphs\\Exploration_diamond.html'
diamonds_csv = pd.read_csv(csv_directory).drop(['id'], axis=1)  # Drop ID column


def row_column_index_creator(index_row_size, index_col_size):
    row_col_list = []
    index_row_size += 1
    index_col_size += 1
    [row_col_list.append([row, column]) for row in range(1, index_row_size) for column in
     range(1, index_col_size)]
    return row_col_list


def figures_to_html(figs, filename):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


def feature_distribution_graph():
    def value_count_extractor(csv_data, feature_name):
        data_feature = csv_data[feature_name]
        unique_feature_count = data_feature.value_counts(normalize=True)  # Count of unique features
        feature_percentage_count = unique_feature_count.values  # Percentage makeup of each unique feature
        feature_data = unique_feature_count.keys()  # Name of data features

        # If data is not a string (int or float)
        if not isinstance(feature_data[0], str):
            bin_list = pd.cut(data_feature, 5, retbins=True)[1]

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

    row_col_index_list = row_column_index_creator(index_row_size=5, index_col_size=2)

    feature_distribution_figure = make_subplots(rows=5, cols=2, subplot_titles=(
        'Carat Distribution', 'Cut Quality Distribution', 'Color Distribution', 'Clarity Distribution',
        'Depth Percent Distribution', 'Table Size Distribution', 'Price mm Distribution', 'Length mm Distribution',
        'Width mm Distribution', 'Depth mm Distribution'))

    for feature, row_col in zip(diamonds_csv.columns, row_col_index_list):
        y_values, range_list = value_count_extractor(diamonds_csv, feature)

        feature_distribution_figure.add_trace(go.Bar(x=range_list, y=y_values, name=feature), row=row_col[0],
                                              col=row_col[1])

    feature_distribution_figure.update_layout(height=2000, width=1300, title_text='Feature Distributions', )

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
    correlation_diamonds_csv = diamonds_csv.select_dtypes(['object'])
    column_headers = correlation_diamonds_csv.columns

    def cramers_v(feature_crosstab_matrix):
        """
        calculate Cramers V statistic for categorical-categorical association.
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
        correlation_value = np.sqrt(phi2corr / min((k_corr - 1), (r_corr - 1)))

        return correlation_value

    def cramer_correlation_extractor_iterator(categorical_features_list):
        categorical_correlation_dataframe = pd.DataFrame(columns=[categorical_features_list], index=[column_headers])
        for first_categorical_feature in categorical_features_list:
            for second_categorical_feature in column_headers:
                confusion_matrix = pd.crosstab(diamonds_csv[first_categorical_feature],
                                               diamonds_csv[second_categorical_feature])
                categorical_correlation_dataframe.loc[second_categorical_feature,
                                                      first_categorical_feature] = cramers_v(confusion_matrix.values)

        return categorical_correlation_dataframe

    categorical_features = ['cut', 'clarity', 'color']
    categorical_feature_dataframe = cramer_correlation_extractor_iterator(categorical_features)

    categorical_correlation_heatmap_figure = go.Figure(
        go.Heatmap(z=categorical_feature_dataframe, x=categorical_features, y=column_headers))
    categorical_correlation_heatmap_figure.update_layout(title_text='Categorical Feature Correlation Heatmap')

    return categorical_correlation_heatmap_figure


def variance_graph():  # Numerical x Numerical
    cleaned_data = diamonds_csv.drop(['price', 'color', 'cut', 'clarity'], axis=1)
    variance_data = np.var(cleaned_data, ddof=1)

    variance_y_list = []
    [variance_y_list.append(variance_datapoint) for variance_datapoint in variance_data]

    variance_figure = go.Figure(data=[go.Bar(x=cleaned_data.columns, y=variance_y_list, name="Variance Data")])
    variance_figure.update_layout(title_text='Feature Variance')

    return variance_figure


def covariance_graph():  # Categorical x Categorical
    cleaned_data = diamonds_csv.drop(['color', 'cut', 'clarity'], axis=1)

    row_col_index_list = row_column_index_creator(index_row_size=7, index_col_size=3)
    covariance_scatter_figure = make_subplots(rows=7, cols=3)

    unique_combinations = []
    column_headers_1 = cleaned_data.columns
    column_headers_2 = cleaned_data.columns

    for feature_1 in column_headers_1:
        for feature_2 in column_headers_2:
            combo = [feature_1, feature_2]
            reversed_combo = combo[::-1]

            if combo == reversed_combo:
                continue

            elif combo not in unique_combinations and reversed_combo not in unique_combinations:
                unique_combinations.append(list(combo))
            else:
                continue

    for combo, rol_col in zip(unique_combinations, row_col_index_list):
        x = cleaned_data[combo[0]]
        y = cleaned_data[combo[1]]
        combo_name = f'{combo[0]} x {combo[1]} y'

        covariance_scatter_figure.add_trace(
            go.Scatter(x=x.head(500),
                       y=y.head(500),
                       name=combo_name,
                       mode='markers'),
            row=rol_col[0], col=rol_col[1])
        covariance_scatter_figure.add_annotation(xref="x domain", yref="y domain", showarrow=False, text=combo_name,
                                                 x=0.5, y=1.2, row=rol_col[0], col=rol_col[1])
    covariance_scatter_figure.update_layout(height=2000, width=1300, title_text='Covariance Scatter Plot')

    return covariance_scatter_figure


def vif_correlation_graph():
    cleaned_data = diamonds_csv.drop(['color', 'cut', 'clarity'], axis=1)

    def calculate_vif(df, features):
        vif, tolerance = {}, {}
        # all the features that you want to examine
        for feature in features:
            # extract all the other features you will regress against
            x = [f for f in features if f != feature]
            x, y = df[x], df[feature]
            # extract r-squared from the fit
            r2 = LinearRegression().fit(x, y).score(x, y)
            # calculate tolerance
            tolerance[feature] = 1 - r2
            # calculate VIF
            vif[feature] = 1 / (tolerance[feature])
        # return VIF DataFrame
        return vif

    numerical_features = ['price', 'width', 'length', 'depth', 'carat', 'depth_percent', 'table']
    vif_data = calculate_vif(cleaned_data, numerical_features)
    vif_data = list(vif_data.values())

    vif_figure = go.Figure(data=[go.Bar(x=numerical_features, y=vif_data, name='VIF Data')])
    vif_figure.update_layout(title_text='VIF Barchart')

    return vif_figure


figure_list = [feature_distribution_graph(), numerical_correlation_map_graph(), categorical_correlation_map_graph(),
               variance_graph(), covariance_graph(), vif_correlation_graph()]
figures_to_html(figs=figure_list, filename=metric_graphs_dir)
