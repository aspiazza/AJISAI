# Data Exploration

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression

csv_directory = fr'D:\Data-Warehouse\Titanic-Data\train.csv'
metric_graphs_dir = fr'..\Model-Graphs&Logs\Model-Data_titanic\Metric-Graphs\Exploration_titanic.html'
df = pd.read_csv(csv_directory)


# Function adds figures to single webpage
def figures_to_html(figs, filename):
    dashboard = open(filename, 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for fig in figs:
        inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
        dashboard.write(inner_html)
    dashboard.write("</body></html>" + "\n")


# Correlation calc for multi-datatypes
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


# Correlation calc for categorical data
def cramers_v(feature_crosstab_matrix):
    """
    calculate Cramers V statistic for categorical-categorical association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328

    Take two features (One of them categorical) and put them in a pd.crosstab()
    function to produce a correlation coefficient
    """
    chi2 = chi2_contingency(feature_crosstab_matrix)[0]
    n = feature_crosstab_matrix.sum()
    phi2 = chi2 / n
    r, k = feature_crosstab_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    correlation_value = np.sqrt(phi2corr / min((k_corr - 1), (r_corr - 1)))

    return correlation_value


def feature_count_graph():
    feature_distribution_figure = make_subplots(rows=3, cols=2)

    # Survival Count
    survived_x = df['Survived'].value_counts().index.tolist()
    survived_y = df['Survived'].value_counts().values.tolist()
    feature_distribution_figure.add_trace(go.Bar(x=survived_x, y=survived_y, name="Survived Count"), row=1, col=1)
    feature_distribution_figure.add_annotation(xref="x domain", yref="y domain", showarrow=False, text="Survived Count",
                                               x=0.5, y=1.2, row=1, col=1)

    # P_Class
    p_class_x = df['Pclass'].value_counts().index.tolist()
    p_class_y = df['Pclass'].value_counts().values.tolist()
    feature_distribution_figure.add_trace(go.Bar(x=p_class_x, y=p_class_y, name="Class Count"), row=1, col=2)
    feature_distribution_figure.add_annotation(xref="x domain", yref="y domain", showarrow=False, text="Class Count",
                                               x=0.5, y=1.2, row=1, col=2)

    # Sex
    sex_x = df['Sex'].value_counts().index.tolist()
    sex_y = df['Sex'].value_counts().values.tolist()
    feature_distribution_figure.add_trace(go.Bar(x=sex_x, y=sex_y, name="Sex Count"), row=2, col=1)
    feature_distribution_figure.add_annotation(xref="x domain", yref="y domain", showarrow=False, text="Sex Count",
                                               x=0.5, y=1.2, row=2, col=1)

    # Age
    y_age_range_list = []
    age_col = df['Age']
    age_bin_list_x = pd.cut(age_col, 5)
    age_bin_list_y = pd.cut(age_col, 5, retbins=True)[1]
    age_x = pd.value_counts(age_bin_list_y).index.tolist()
    age_y = pd.value_counts(age_bin_list_x).values.tolist()

    for index in range(1, 6):
        if index == 5:  # If reach end of index
            break
        else:
            y_age_range_list.append(f"{('{:.3f}'.format(age_x[index]))} - {('{:.3f}'.format(age_x[index + 1]))}")

    feature_distribution_figure.add_trace(go.Bar(x=y_age_range_list, y=age_y, name="Age Count"), row=2, col=2)
    feature_distribution_figure.add_annotation(xref="x domain", yref="y domain", showarrow=False, text="Age Count",
                                               x=0.5, y=1.2, row=2, col=2)

    # Fare
    y_fare_range_list = []
    fare_col = df['Fare']
    fare_bin_list_x = pd.cut(fare_col, 5)
    fare_bin_list_y = pd.cut(fare_col, 5, retbins=True)[1]
    fare_x = pd.value_counts(fare_bin_list_y).index.tolist()
    fare_y = pd.value_counts(fare_bin_list_x).values.tolist()

    for index in range(1, 6):
        if index == 5:
            break
        else:
            y_fare_range_list.append(f"{('{:.3f}'.format(fare_x[index]))} - {('{:.3f}'.format(fare_x[index + 1]))}")

    feature_distribution_figure.add_trace(go.Bar(x=y_fare_range_list, y=fare_y, name="fare Count"), row=3, col=1)
    feature_distribution_figure.add_annotation(xref="x domain", yref="y domain", showarrow=False, text="fare Count",
                                               x=0.5, y=1.2, row=3, col=1)

    feature_distribution_figure.update_layout(height=2000, width=1300, title_text='Feature Distributions')
    return feature_distribution_figure


# Correlation between numerical features
def num_correlation_map_graph():
    cleaned_titanic_csv = df.select_dtypes(['number'])
    column_headers = cleaned_titanic_csv.columns
    features_correlation = cleaned_titanic_csv.corr()

    correlation_heatmap_figure = go.Figure(
        go.Heatmap(z=features_correlation, x=column_headers, y=column_headers))
    correlation_heatmap_figure.update_layout(title_text='Numerical Feature Correlation Heatmap')

    return correlation_heatmap_figure


# Correlation between numerical features
def cat_correlation_map_graph():
    cleaned_titanic_csv = df.select_dtypes(['object'])
    column_headers = cleaned_titanic_csv.columns
    categorical_correlation_dataframe = pd.DataFrame(columns=[column_headers],
                                                     index=[column_headers])  # Create dataframe

    for first_categorical_feature in column_headers:
        for second_categorical_feature in column_headers:
            confusion_matrix = pd.crosstab(cleaned_titanic_csv[first_categorical_feature],
                                           cleaned_titanic_csv[second_categorical_feature])
            print(confusion_matrix.values)
            exit()

            categorical_correlation_dataframe.loc[second_categorical_feature,
                                                  first_categorical_feature] = cramers_v(confusion_matrix.values)
    print(categorical_correlation_dataframe.head())
    exit()
    categorical_correlation_heatmap_figure = go.Figure(
        go.Heatmap(z=categorical_correlation_dataframe, x=column_headers, y=column_headers))
    categorical_correlation_heatmap_figure.update_layout(title_text='Categorical Feature Correlation Heatmap')

    return categorical_correlation_heatmap_figure


figures_to_html(figs=[feature_count_graph(), num_correlation_map_graph(), cat_correlation_map_graph()],
                filename=metric_graphs_dir)
