# Data Visualization
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from icecream import ic


class DataVis:
    def __init__(self, history, model_name):
        self.graph_storage_directory = f'C:\\Users\\17574\\PycharmProjects\\Kraken\\Kraken_Project\\AJISAI-Project\\Model-Graphs&Logs\\Model-Data_{model_name}\\Metric-Graphs\\'
        self.epoch_list = history.epoch
        self.accuracy_list = history.history['accuracy']
        self.loss_list = history.history['loss']
        self.val_accuracy_list = history.history['val_accuracy']
        self.val_loss_list = history.history['val_loss']

    def loss_graph(self):
        self.loss_fig = go.Figure()

        ic(self.epoch_list)
        ic(self.accuracy_list)
        ic(self.loss_list)

        self.loss_fig.add_traces(
            [go.Scatter(x=self.epoch_list, y=self.val_loss_list,
                        mode='lines',
                        name='Validation Loss'),
             go.Scatter(x=self.epoch_list, y=self.loss_list,
                        mode='lines',
                        name='Loss')]
        )
        self.loss_fig.update_layout(
            font_color='black',
            title_font_color='black',
            title='Loss Curve'
        )
        self.loss_fig.show()

        # ic(self.loss_fig)  # TODO: plot_model

    def subplot_creation(self):
        metric_figure = make_subplots(
            rows=1, cols=2,
            specs=[[{'rowspan': 1}, {}]],
            subplot_titles='Dog Cat Loss Graph')

        metric_figure.add_traces(self.loss_fig)  # TODO: Doesn't like data

        plotly.offline.plot(metric_figure,
                            filename=f'{self.graph_storage_directory}Metric_graph_dog_cat.html',
                            auto_open=False)
