# Data Visualization
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class DataVisualization:
    def __init__(self, metric_data, metric_dir):

        # If file type is equal to tensorflow history
        if str(type(metric_data)) == "<class 'tensorflow.python.keras.callbacks.History'>":
            self.epoch_list = metric_data.epoch
            metric_data = metric_data.history
            self.last_auc_score = metric_data['auc'][-1]
            self.val_last_auc_score = metric_data['val_auc'][-1]

        # If file type is equal to CSV dataframe
        elif str(type(metric_data)) == "<class 'pandas.core.frame.DataFrame'>":
            self.epoch_list = metric_data['epoch']
            self.last_auc_score = metric_data['auc'].iloc[-1]
            self.val_last_auc_score = metric_data['val_auc'].iloc[-1]

        self.metric_dir = metric_dir
        self.subplot_name_list = []
        self.figure_xaxes_list = []
        self.figure_yaxes_list = []
        self.subplot_list = []

        self.accuracy_list = metric_data['accuracy']
        self.loss_list = metric_data['loss']
        self.recall = metric_data['recall']  # true_positives
        self.precision = metric_data['precision']
        self.true_positives = metric_data['true_positives']
        self.true_negatives = metric_data['true_negatives']
        self.false_positives = metric_data['false_positives']
        self.false_negatives = metric_data['false_negatives']

        self.val_accuracy_list = metric_data['val_accuracy']
        self.val_loss_list = metric_data['val_loss']
        self.val_recall = metric_data['val_recall']
        self.val_precision = metric_data['val_precision']
        self.val_true_positives = metric_data['val_true_positives']
        self.val_true_negatives = metric_data['val_true_negatives']
        self.val_false_positives = metric_data['val_false_positives']
        self.val_false_negatives = metric_data['val_false_negatives']

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

    #def confusion_matrx(self):


    def subplot_creation(self, row_size, col_size):

        def row_column_index_creator(index_row_size, index_col_size):
            row_col_index_list = []
            index_row_size += 1
            index_col_size += 1
            [row_col_index_list.append([row, column]) for row in range(1, index_row_size) for column
             in range(1, index_col_size)]
            return row_col_index_list

        def axes_title_creator(xaxes_list, yaxes_list):
            x_y_axes = []
            for (xaxes, yaxes) in zip(xaxes_list, yaxes_list):
                x_y_axes.append([xaxes, yaxes])
            return x_y_axes

        row_col_index_list = row_column_index_creator(row_size, col_size)
        x_y_axes = axes_title_creator(self.figure_xaxes_list, self.figure_yaxes_list)
        metric_subplot = make_subplots(rows=row_size, cols=col_size, subplot_titles=self.subplot_name_list)

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
                            filename=f'{self.metric_dir}_metrics.html',
                            auto_open=False)
