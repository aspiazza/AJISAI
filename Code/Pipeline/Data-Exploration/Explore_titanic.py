# Data Exploration

from numpy import sqrt, var
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
    age_col = df['Age'].dropna()
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
    fare_col = df['Fare'].dropna()
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
def numerical_correlation_map_graph():
    numerical_titanic_csv = df.drop(['PassengerId', 'Name'], axis=1)  # .dropna()
    numerical_features_correlation = numerical_titanic_csv.corr()
    column_headers = numerical_titanic_csv.columns

    numerical_correlation_heatmap_figure = go.Figure(
        go.Heatmap(z=numerical_features_correlation, x=column_headers, y=column_headers))
    numerical_correlation_heatmap_figure.update_layout(title_text='Numerical Feature Correlation Heatmap')

    return numerical_correlation_heatmap_figure


figures_to_html(figs=[feature_count_graph()], filename=metric_graphs_dir)
