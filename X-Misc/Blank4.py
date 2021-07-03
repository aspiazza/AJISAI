
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
from sklearn.datasets import make_classification
from icecream import ic

X, y = make_classification(n_samples=500, random_state=0)

model = LogisticRegression()
model.fit(X, y)
y_score = model.predict_proba(X)[:, 1]

precision, recall, thresholds = precision_recall_curve(y, y_score)

ic(precision)
ic(recall)


# Storing code


# Data Visualization WIP
'''
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from icecream import ic
import pandas as pd


class DataVis:
    def __init__(self, history):
        self.epoch_list = history['epoch']
        self.subplot_name_list = []
        self.subplot_layout_list = []
        self.subplot_list = []

        self.accuracy_list = history['accuracy']
        self.loss_list = history['loss']
        self.recall = history['recall']  # true_positives
        self.precision = history['precision']
        self.false_positives = history['false_positives']
        self.true_negatives = history['true_negatives']
        self.false_negatives = history['false_negatives']
        self.last_auc_score = history['auc'].iloc[-1]

        self.val_accuracy_list = history['val_accuracy']
        self.val_loss_list = history['val_loss']
        self.val_recall = history['val_recall']
        self.val_precision = history['val_precision']
        self.val_false_positives = history['val_false_positives']
        self.val_true_negatives = history['val_true_negatives']
        self.val_false_negatives = history['val_false_negatives']
        self.val_last_auc_score = history['val_auc'].iloc[-1]

    def loss_graph(self):
        loss_plots = [go.Scatter(x=self.epoch_list,
                                 y=self.loss_list,
                                 mode='lines',
                                 name='Loss',
                                 line=dict(width=4)),
                      go.Scatter(x=self.epoch_list,
                                 y=self.val_loss_list,
                                 mode='lines',
                                 name='Validation Loss',
                                 line=dict(width=4))]
        loss_layout = dict(font_color='black',
                           title_font_color='black',
                           title=dict(text='Loss Graph',
                                      font_size=30),
                           xaxis_title=dict(text='Epochs',
                                            font_size=25),
                           yaxis_title=dict(text='Loss',
                                            font_size=25),
                           legend=dict(font_size=15))

        self.loss_figure = go.Figure(data=loss_plots)
        self.subplot_name_list.append('Loss Graph')
        self.subplot_layout_list.append(loss_layout)
        self.subplot_list.append(self.loss_figure)

    def error_rate_graph(self):

        def error_rate_computation(accuracy):
            error_rate_list = []
            for accuracy_instance in accuracy:
                error_rate_list.append(1 - accuracy_instance)
            return error_rate_list

        train_error_rate = error_rate_computation(self.accuracy_list)
        valid_error_rate = error_rate_computation(self.val_accuracy_list)

        error_rate_plots = [go.Scatter(x=self.epoch_list,
                                       y=train_error_rate,
                                       mode='lines',
                                       name='Error Rate',
                                       line=dict(width=4)),
                            go.Scatter(x=self.epoch_list,
                                       y=valid_error_rate,
                                       mode='lines',
                                       name='Validation Error Rate',
                                       line=dict(width=4))]
        error_rate_layout = dict(font_color='black',
                                 title_font_color='black',
                                 title=dict(text='Error Rate Graph',
                                            font_size=30),
                                 xaxis_title=dict(text='Epochs',
                                                  font_size=25),
                                 yaxis_title=dict(text='Error Rate',
                                                  font_size=25),
                                 legend=dict(font_size=15))

        self.error_rate_figure = go.Figure(data=error_rate_plots)
        self.subplot_name_list.append('Error Rate Graph')
        self.subplot_layout_list.append(error_rate_layout)
        self.subplot_list.append(self.error_rate_figure)

    def recall_graph(self):
        recall_plots = [go.Scatter(x=self.epoch_list,
                                   y=self.recall,
                                   mode='lines',
                                   name='Recall',
                                   line=dict(width=4)),
                        go.Scatter(x=self.epoch_list,
                                   y=self.val_recall,
                                   mode='lines',
                                   name='Validation Recall',
                                   line=dict(width=4))]
        recall_layout = dict(font_color='black',
                             title_font_color='black',
                             title=dict(text='Recall Graph',
                                        font_size=30),
                             xaxis_title=dict(text='Epochs',
                                              font_size=25),
                             yaxis_title=dict(text='Recall',
                                              font_size=25), )

        self.recall_figure = go.Figure(data=recall_plots)
        self.subplot_name_list.append('Recall Graph')
        self.subplot_layout_list.append(recall_layout)
        self.subplot_list.append(self.recall_figure)

    def precision_graph(self):
        precision_plots = [go.Scatter(x=self.epoch_list,
                                      y=self.precision,
                                      mode='lines',
                                      name='Precision',
                                      line=dict(width=4)),
                           go.Scatter(x=self.epoch_list,
                                      y=self.val_precision,
                                      mode='lines',
                                      name='Validation Precision',
                                      line=dict(width=4))]
        precision_layout = dict(font_color='black',
                                title_font_color='black',
                                title=dict(text='Precision Graph',
                                           font_size=30),
                                xaxis_title=dict(text='Epochs',
                                                 font_size=25),
                                yaxis_title=dict(text='Precision',
                                                 font_size=25),
                                legend=dict(font_size=15))

        self.precision_figure = go.Figure(data=precision_plots)
        self.subplot_name_list.append('Precision Graph')
        self.subplot_layout_list.append(precision_layout)
        self.subplot_list.append(self.precision_figure)

    def f1_graph(self):
        def f1_score_computation(precision, recall):
            f1_score_list = []
            for (precision_score, recall_score) in zip(precision, recall):
                f1_score_list.append(2 * ((precision_score * recall_score) / (precision_score + recall_score)))
            return f1_score_list

        f1_scores = f1_score_computation(self.precision, self.recall)
        val_f1_scores = f1_score_computation(self.val_precision, self.val_recall)

        f1_plots = [go.Scatter(x=self.epoch_list,
                               y=f1_scores,
                               mode='lines',
                               name='F1 Score',
                               line=dict(width=4)),
                    go.Scatter(x=self.epoch_list,
                               y=val_f1_scores,
                               mode='lines',
                               name='Validation F1 Score',
                               line=dict(width=4))]
        f1_layout = dict(font_color='black',
                         title_font_color='black',
                         title=dict(text='F1 Graph',
                                    font_size=30),
                         xaxis_title=dict(text='Epochs',
                                          font_size=25),
                         yaxis_title=dict(text='F1 Score',
                                          font_size=25),
                         legend=dict(font_size=15))

        self.f1_figure = go.Figure(data=f1_plots)
        self.subplot_name_list.append('F1 Graph')
        self.subplot_layout_list.append(f1_layout)
        self.subplot_list.append(self.f1_figure)

    def subplot_creation(self, context, row_size, col_size, model_name):

        metric_figure = make_subplots(rows=row_size, cols=col_size, subplot_titles=self.subplot_name_list)

        row_col_index_list = []  # TODO: Find a way to move over layouts
        row_size -= 1
        col_size += 1
        for row_index in range(col_size):
            row_index += 1
            for col_index in range(row_size):
                col_index += 1
                row_col_index_list.append(f'{row_index},{col_index}')

        for plot, row_col in zip(self.subplot_list, row_col_index_list):
            row_col = row_col.split(',')
            row_index = int(row_col[0])
            col_index = int(row_col[1])
            for trace in plot.data:
                metric_figure.append_trace(trace, row=row_index, col=col_index)

        for (figure, fig_layout) in zip(self.subplot_list, self.subplot_layout_list):
            figure.update_layout(fig_layout)
            metric_figure.show()
            exit()

        #metric_figure.show()

        # plotly.offline.plot(metric_subplot,
        #                     filename=f'{context}_metric_graph_{model_name}.html',
        #                     auto_open=False)


dir = 'C:\\Users\\17574\\PycharmProjects\\Kraken\\AJISAI-Project\\Model-Graphs&Logs\\Model-Data_dog_cat\\Logs'
csv_file = 'dog_cat_13-48-44_training_metrics.csv'
csv = pd.read_csv(f'{dir}\\{csv_file}')
test = DataVis(csv)
test.loss_graph()
test.error_rate_graph()
test.recall_graph()
test.precision_graph()
test.f1_graph()
test.subplot_creation('Training', row_size=3, col_size=2, model_name='dog_cat')

'''