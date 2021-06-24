# Data Visualization
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class TrainingDataVis:
    def __init__(self, history, model_name):
        self.graph_storage_directory = f'C:\\Users\\17574\\PycharmProjects\\Kraken\\Kraken_Project\\AJISAI-Project\\Model-Graphs&Logs\\Model-Data_{model_name}\\Metric-Graphs\\'
        self.epoch_list = history['epoch']

        self.accuracy_list = history.history['accuracy']
        self.loss_list = history.history['loss']
        self.recall = history.history['recall']  # true_positives
        self.precision = history.history['precision']
        self.true_positives = history.history['true_positives']
        self.true_negatives = history.history['true_negatives']
        self.false_positives = history.history['false_positives']
        self.false_negatives = history.history['false_negatives']
        self.last_auc_score = history.history['auc'].iloc[-1]

        self.val_accuracy_list = history.history['val_accuracy']
        self.val_loss_list = history.history['val_loss']
        self.val_recall = history.history['val_recall']
        self.val_precision = history.history['val_precision']
        self.val_true_positives = history.history['val_true_positives']
        self.val_true_negatives = history.history['val_true_negatives']
        self.val_false_positives = history.history['val_false_positives']
        self.val_false_negatives = history.history['val_false_negatives']
        self.val_last_auc_score = history.history['val_auc'].iloc[-1]

    def loss_graph(self):
        self.loss_graph = go.Figure()
        self.loss_graph.add_traces(
            [go.Scatter(x=self.epoch_list,
                        y=self.loss_list,
                        mode='lines',
                        name='Loss',
                        line=dict(width=4)),
             go.Scatter(x=self.epoch_list,
                        y=self.val_loss_list,
                        mode='lines',
                        name='Validation Loss',
                        line=dict(width=4))])

        self.loss_graph.update_layout(
            font_color='black',
            title_font_color='black',
            title=dict(text='Loss Graph',
                       font_size=30),
            xaxis_title=dict(text='Epochs',
                             font_size=25),
            yaxis_title=dict(text='Loss',
                             font_size=25),
            legend=dict(font_size=15)
        )
        return self.loss_graph

    def error_rate_graph(self):
        def error_rate_computation(accuracy):
            error_rate_list = []
            for accuracy_instance in accuracy:
                error_rate_list.append(1 - accuracy_instance)
            return error_rate_list

        train_error_rate = error_rate_computation(self.accuracy_list)
        valid_error_rate = error_rate_computation(self.val_accuracy_list)

        self.error_rate_figure = go.Figure()
        self.error_rate_figure.add_traces(
            [go.Scatter(x=self.epoch_list,
                        y=train_error_rate,
                        mode='lines',
                        name='Error Rate',
                        line=dict(width=4)),
             go.Scatter(x=self.epoch_list,
                        y=valid_error_rate,
                        mode='lines',
                        name='Validation Error Rate',
                        line=dict(width=4))])

        self.error_rate_figure.update_layout(
            font_color='black',
            title_font_color='black',
            title=dict(text='Error Rate Graph',
                       font_size=30),
            xaxis_title=dict(text='Epochs',
                             font_size=25),
            yaxis_title=dict(text='Error Rate',
                             font_size=25),
            legend=dict(font_size=15)
        )
        return self.error_rate_figure

    def recall_graph(self):
        self.recall_figure = go.Figure()
        self.recall_figure.add_traces(
            [go.Scatter(x=self.epoch_list,
                        y=self.recall,
                        mode='lines',
                        name='Recall',
                        line=dict(width=4)),
             go.Scatter(x=self.epoch_list,
                        y=self.val_recall,
                        mode='lines',
                        name='Validation Recall',
                        line=dict(width=4))])

        self.recall_figure.update_layout(
            font_color='black',
            title_font_color='black',
            title=dict(text='Recall Graph',
                       font_size=30),
            xaxis_title=dict(text='Epochs',
                             font_size=25),
            yaxis_title=dict(text='Recall',
                             font_size=25),
            legend=dict(font_size=15)
        )
        self.recall_figure.show()
        return self.recall_figure

    def precision_graph(self):
        self.precision_figure = go.Figure()
        self.precision_figure.add_traces(
            [go.Scatter(x=self.epoch_list,
                        y=self.precision,
                        mode='lines',
                        name='Precision',
                        line=dict(width=4)),
             go.Scatter(x=self.epoch_list,
                        y=self.val_precision,
                        mode='lines',
                        name='Validation Precision',
                        line=dict(width=4))])

        self.precision_figure.update_layout(
            font_color='black',
            title_font_color='black',
            title=dict(text='Precision Graph',
                       font_size=30),
            xaxis_title=dict(text='Epochs',
                             font_size=25),
            yaxis_title=dict(text='Precision',
                             font_size=25),
            legend=dict(font_size=15)
        )
        self.precision_figure.show()
        return self.precision_figure

    def f1_graph(self):
        def f1_score_computation(precision, recall):
            f1_score_list = []
            for (precision_score, recall_score) in zip(precision, recall):
                f1_score_list.append(2 * ((precision_score * recall_score) / (precision_score + recall_score)))
            return f1_score_list

        f1_scores = f1_score_computation(self.precision, self.recall)
        val_f1_scores = f1_score_computation(self.val_precision, self.val_recall)

        self.f1_figure = go.Figure()
        self.f1_figure.add_traces(
            [go.Scatter(x=self.epoch_list,
                        y=f1_scores,
                        mode='lines',
                        name='F1 Score',
                        line=dict(width=4)),
             go.Scatter(x=self.epoch_list,
                        y=val_f1_scores,
                        mode='lines',
                        name='Validation F! Score',
                        line=dict(width=4))])

        self.f1_figure.update_layout(
            font_color='black',
            title_font_color='black',
            title=dict(text='F1 Graph',
                       font_size=30),
            xaxis_title=dict(text='Epochs',
                             font_size=25),
            yaxis_title=dict(text='F1 Score',
                             font_size=25),
            legend=dict(font_size=15)
        )
        self.f1_figure.show()
        return self.f1_figure

    def subplot_creation(self):
        metric_figure = make_subplots(
            subplot_titles='Dog Cat Loss Graph')

        metric_figure.add_traces(self.precision_graph)

        plotly.offline.plot(metric_figure,
                            filename=f'Metric_graph_dog_cat.html',
                            auto_open=False)


class TestingDataVis:
    def __init__(self):
        return
