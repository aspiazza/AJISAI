# Data Visualization
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from icecream import ic


class DataVisualization:
    def __init__(self, training_information, metric_dir):
        self.metric_dir = metric_dir
        self.subplot_name_list = []
        self.figure_xaxes_list = []
        self.figure_yaxes_list = []
        self.subplot_list = []

        if str(type(training_information)) == "<class 'tensorflow.python.keras.callbacks.History'>":
            self.epoch_list = training_information.epoch
            training_information = training_information.history
            self.last_auc_score = training_information['auc'][-1]
            self.val_last_auc_score = training_information['val_auc'][-1]

        elif str(type(training_information)) == "<class 'pandas.core.frame.DataFrame'>":
            self.epoch_list = training_information['epoch']
            self.last_auc_score = training_information['auc'].iloc[-1]
            self.val_last_auc_score = training_information['val_auc'].iloc[-1]

        self.accuracy_list = training_information['accuracy']
        self.loss_list = training_information['loss']
        self.recall = training_information['recall']  # true_positives
        self.precision = training_information['precision']
        self.true_positives = training_information['true_positives']
        self.true_negatives = training_information['true_negatives']
        self.false_positives = training_information['false_positives']
        self.false_negatives = training_information['false_negatives']

        self.val_accuracy_list = training_information['val_accuracy']
        self.val_loss_list = training_information['val_loss']
        self.val_recall = training_information['val_recall']
        self.val_precision = training_information['val_precision']
        self.val_true_positives = training_information['val_true_positives']
        self.val_true_negatives = training_information['val_true_negatives']
        self.val_false_positives = training_information['val_false_positives']
        self.val_false_negatives = training_information['val_false_negatives']

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

        self.loss_figure = go.Figure(data=loss_plots)
        self.subplot_name_list.append('Loss Graph')
        self.figure_xaxes_list.append("Epochs")
        self.figure_yaxes_list.append("Loss")
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

        self.error_rate_figure = go.Figure(data=error_rate_plots)
        self.subplot_name_list.append('Error Rate Graph')
        self.figure_xaxes_list.append("Epochs")
        self.figure_yaxes_list.append("Error Rate")
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

        self.recall_figure = go.Figure(data=recall_plots)
        self.subplot_name_list.append('Recall Graph')
        self.figure_xaxes_list.append("Epochs")
        self.figure_yaxes_list.append("Recall")
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

        self.precision_figure = go.Figure(data=precision_plots)
        self.subplot_name_list.append('Precision Graph')
        self.figure_xaxes_list.append("Epochs")
        self.figure_yaxes_list.append("Precision")
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

        self.f1_figure = go.Figure(data=f1_plots)
        self.subplot_name_list.append('F1 Graph')
        self.figure_xaxes_list.append("Epochs")
        self.figure_yaxes_list.append("F1 Score")
        self.subplot_list.append(self.f1_figure)

    def subplot_creation(self, context, row_size, col_size):

        metric_subplot = make_subplots(rows=row_size, cols=col_size, subplot_titles=self.subplot_name_list)

        def row_index_creator(row_size, col_size):
            row_col_index_list = []
            row_size -= 1
            col_size += 1
            for row_index in range(col_size):
                row_index += 1
                for col_index in range(row_size):
                    col_index += 1
                    row_col_index_list.append([row_index, col_index])
            return row_col_index_list

        def axes_title_creator(xaxes_list, yaxes_list):
            x_y_axes = []
            for (xaxes, yaxes) in zip(xaxes_list, yaxes_list):
                x_y_axes.append([xaxes, yaxes])
            return x_y_axes

        row_col_index_list = row_index_creator(row_size, col_size)
        x_y_axes = axes_title_creator(self.figure_xaxes_list, self.figure_yaxes_list)

        for plot, row_col, x_y_ax in zip(self.subplot_list, row_col_index_list, x_y_axes):
            x_axes = x_y_ax[0]
            y_axes = x_y_ax[1]
            row_index = row_col[0]
            col_index = row_col[1]

            metric_subplot.update_xaxes(title_text=x_axes, row=row_index, col=col_index)
            metric_subplot.update_yaxes(title_text=y_axes, row=row_index, col=col_index)

            for trace in plot.data:
                metric_subplot.append_trace(trace, row=row_index, col=col_index)

        plotly.offline.plot(metric_subplot,
                            filename=f'{self.metric_dir}_{context}_metrics.html',
                            auto_open=False)
