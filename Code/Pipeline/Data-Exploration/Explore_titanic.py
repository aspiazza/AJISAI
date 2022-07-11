# Data Exploration

from numpy import sqrt, var
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression

csv_directory = fr'D:\Data-Warehouse\Titanic-Data\train.csv'
metric_graphs_dir = fr'..\Model-Graphs&Logs\Model-Data_titanic\Metric-Graphs\Exploration_titanic.html'


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
    df = pd.read_csv(csv_directory)

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
    feature_distribution_figure.add_annotation(xref="x domain", yref="y domain", showarrow=False, text="Survived Count",
                                               x=0.5, y=1.2, row=1, col=2)

    # Sex
    sex_x = df['Sex'].value_counts().index.tolist()
    sex_y = df['Sex'].value_counts().values.tolist()
    feature_distribution_figure.add_trace(go.Bar(x=sex_x, y=sex_y, name="Sex Count"), row=2, col=1)
    feature_distribution_figure.add_annotation(xref="x domain", yref="y domain", showarrow=False, text="Sex Count",
                                               x=0.5, y=1.2, row=2, col=1)

    # Age
    age_x = df['Age'].dropna()
    age_bin_list = pd.cut(age_x, 5, retbins=True)
    print(age_bin_list[-1])


    # Fare

    feature_distribution_figure.update_layout(height=2000, width=1300, title_text='Feature Distributions')


feature_count_graph()
