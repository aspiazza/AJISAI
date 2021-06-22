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
        self.true_positives = history.history['true_positives']
        self.false_positives = history.history['false_positives']
        self.true_negatives = history.history['true_negatives']
        self.false_negatives = history.history['false_negatives']
        self.last_auc_score = history.history['auc'].iloc[-1]

        self.val_accuracy_list = history.history['val_accuracy']
        self.val_loss_list = history.history['val_loss']
        self.val_true_positives = history.history['val_true_positives']
        self.val_false_positives = history.history['val_false_positives']
        self.val_true_negatives = history.history['val_true_negatives']
        self.val_false_negatives = history.history['val_false_negatives']
        self.val_last_auc_score = history.history['val_auc'].iloc[-1]

    def loss_graph(self):

        self.loss_figure = go.Figure()
        self.loss_figure.add_traces(
            [go.Scatter(x=self.epoch_list,
                        y=self.val_loss_list,
                        mode='lines',
                        name='Validation Loss',
                        line=dict(width=4)),
             go.Scatter(x=self.epoch_list,
                        y=self.loss_list,
                        mode='lines',
                        name='Loss',
                        line=dict(width=4))])

        self.loss_figure.update_layout(
            font_color='black',
            title_font_color='black',
            title=dict(text='Loss Curve',
                       font_size=30),
            xaxis_title=dict(text='Epochs',
                             font_size=25),
            yaxis_title=dict(text='loss',
                             font_size=25),
            legend=dict(font_size=15)
        )
        return self.loss_figure

    def roc_curve(self):  # TODO: Move to testing class?

        def true_positive_rate(false_negatives, true_positives):
            true_positive_rates = []
            for (fn, tp) in zip(false_negatives, true_positives):
                true_positive_rates.append(tp / (tp + fn))
            return true_positive_rates

        def false_positive_rate(false_positives, true_negatives):
            false_positive_rates = []
            for (fp, tn) in zip(false_positives, true_negatives):
                false_positive_rates.append(fp / (fp + tn))
            return false_positive_rates

        x_true_positive_rate = true_positive_rate(self.false_negatives, self.true_positives)
        y_false_positive_rate = false_positive_rate(self.false_positives, self.true_negatives)
        x_val_true_positive_rate = true_positive_rate(self.val_false_negatives, self.val_false_positives)
        y_val_false_positive_rate = false_positive_rate(self.val_false_positives, self.val_true_negatives)

        self.roc_figure = go.Figure()
        self.roc_figure.add_traces(
            [go.Scatter(x=x_true_positive_rate,
                        y=y_false_positive_rate,
                        mode='lines',
                        name='ROC Curve'),
             go.Scatter(x=x_val_true_positive_rate,
                        y=y_val_false_positive_rate,
                        mode='lines',
                        name='Validation ROC Curve')])

        self.roc_figure.update_layout(
            font_color='black',
            title_font_color='black',
            title=dict(text=f'ROC Curve  (Last AUC: {self.last_auc_score:.4f}'),
            font_size=30,
            xaxis_title=dict(text='False Positive Rate',
                             font_size=25),
            yaxis_title=dict(text='True Positive Rate',
                             font_size=25),
            legend=dict(font_size=15))

        self.roc_figure.add_shape(type='line', line=dict(dash='dash'),
                                  x0=0, x1=1, y0=0, y1=1)

        self.roc_figure.update_yaxes(scaleanchor="x", scaleratio=1)

        self.roc_figure.update_xaxes(constrain='domain')

        return self.roc_figure

    def subplot_creation(self):
        metric_figure = make_subplots(
            subplot_titles='Dog Cat Loss Graph')

        metric_figure.add_traces(self.loss_figure)

        plotly.offline.plot(metric_figure,
                            filename=f'Metric_graph_dog_cat.html',
                            auto_open=False)


class TestingDataVis:
    def __init__(self):
        return
