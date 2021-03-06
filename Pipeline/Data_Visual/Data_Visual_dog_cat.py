# Data Visualization

from plotly.subplots import make_subplots
from plotly import offline
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np


class TrainingDataVisualization:
    def __init__(self, metric_data, metric_dir):

        # If file type is equal to tensorflow history
        if str(type(metric_data)) == "<class 'tensorflow.python.keras.callbacks.History'>":
            self.epoch_list = metric_data.epoch

            metric_data = metric_data.history

            self.last_auc_score = metric_data['auc'][-1]
            self.val_last_auc_score = metric_data['val_auc'][-1]

            self.accuracy_list = metric_data['accuracy']
            self.loss_list = metric_data['loss']
            self.recall = metric_data['recall']
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

            self.last_true_negatives = self.true_negatives[-1]
            self.last_false_negatives = self.false_negatives[-1]
            self.last_true_positives = self.true_positives[-1]
            self.last_false_positives = self.false_positives[-1]

            self.last_val_true_negatives = self.val_true_negatives[-1]
            self.last_val_false_negatives = self.val_false_negatives[-1]
            self.last_val_true_positives = self.val_true_positives[-1]
            self.last_val_false_positives = self.val_false_positives[-1]

        # If file type is equal to CSV dataframe
        elif str(type(metric_data)) == "<class 'pandas.core.frame.DataFrame'>":
            self.epoch_list = metric_data['epoch']
            self.last_auc_score = metric_data['auc'].iloc[-1]
            self.val_last_auc_score = metric_data['val_auc'].iloc[-1]

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

            self.last_true_negatives = self.true_negatives.iloc[-1]
            self.last_false_negatives = self.false_negatives.iloc[-1]
            self.last_true_positives = self.true_positives.iloc[-1]
            self.last_false_positives = self.false_positives.iloc[-1]

            self.last_val_true_negatives = self.val_true_negatives.iloc[-1]
            self.last_val_false_negatives = self.val_false_negatives.iloc[-1]
            self.last_val_true_positives = self.val_true_positives.iloc[-1]
            self.last_val_false_positives = self.val_false_positives.iloc[-1]

        self.metric_dir = metric_dir
        self.subplot_name_list = []
        self.figure_x_axes_list = []
        self.figure_y_axes_list = []
        self.subplot_list = []

        self.loss_figure = None
        self.error_rate_figure = None
        self.precision_figure = None
        self.recall_figure = None
        self.recall_figure = None
        self.f1_figure = None
        self.fpr_figure = None
        self.fnr_figure = None
        self.tpr_figure = None
        self.tnr_figure = None

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
        self.figure_x_axes_list.append("Epochs")
        self.figure_y_axes_list.append("Loss")
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
        self.figure_x_axes_list.append("Epochs")
        self.figure_y_axes_list.append("Error Rate")
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
        self.figure_x_axes_list.append("Epochs")
        self.figure_y_axes_list.append("Recall")
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
        self.figure_x_axes_list.append("Epochs")
        self.figure_y_axes_list.append("Precision")
        self.subplot_list.append(self.precision_figure)

    def f1_graph(self):

        def f1_score_computation(precision, recall):
            f1_score_list = []
            for (precision_score, recall_score) in zip(precision, recall):
                try:
                    f1_score_list.append(2 * ((precision_score * recall_score) / (precision_score + recall_score)))
                except ZeroDivisionError:  # If F1 score instance is equal to 0
                    f1_score_list.append(np.nan)
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
        self.figure_x_axes_list.append("Epochs")
        self.figure_y_axes_list.append("F1 Score")
        self.subplot_list.append(self.f1_figure)

    def false_positive_graph(self):

        def false_positive_computation(false_positives, true_negatives):
            fp_rate_list = []
            for (fp_score, tn_score) in zip(false_positives, true_negatives):
                fp_rate_list.append(fp_score / (fp_score + tn_score))
            return fp_rate_list

        false_positive_rate = false_positive_computation(self.false_positives, self.true_negatives)
        val_false_positive_rate = false_positive_computation(self.val_false_positives, self.val_true_negatives)

        fpr_plots = [go.Scatter(x=self.epoch_list,
                                y=false_positive_rate,
                                mode='lines',
                                name='False Positive Rate',
                                line=dict(width=4)),
                     go.Scatter(x=self.epoch_list,
                                y=val_false_positive_rate,
                                mode='lines',
                                name='Validation False Positive Rate',
                                line=dict(width=4))]

        self.fpr_figure = go.Figure(data=fpr_plots)
        self.subplot_name_list.append('False Positive Rate Graph')
        self.figure_x_axes_list.append("Epochs")
        self.figure_y_axes_list.append("False Positive Rate")
        self.subplot_list.append(self.fpr_figure)

    def false_negative_graph(self):

        def false_negative_computation(false_negatives, true_positives):
            fn_rate_list = []
            for (fn_score, tp_score) in zip(false_negatives, true_positives):
                fn_rate_list.append(fn_score / (fn_score + tp_score))
            return fn_rate_list

        false_negative_rate = false_negative_computation(self.false_negatives, self.true_positives)
        val_false_negative_rate = false_negative_computation(self.val_false_negatives, self.val_true_positives)

        fnr_plots = [go.Scatter(x=self.epoch_list,
                                y=false_negative_rate,
                                mode='lines',
                                name='False Negative Rate',
                                line=dict(width=4)),
                     go.Scatter(x=self.epoch_list,
                                y=val_false_negative_rate,
                                mode='lines',
                                name='Validation False Negative Rate',
                                line=dict(width=4))]

        self.fnr_figure = go.Figure(data=fnr_plots)
        self.subplot_name_list.append('False Negative Rate Graph')
        self.figure_x_axes_list.append("Epochs")
        self.figure_y_axes_list.append("False Negative Rate")
        self.subplot_list.append(self.fnr_figure)

    def true_positive_graph(self):

        def true_positive_computation(true_positives, false_negatives):
            tp_rate_list = []
            for (tp_score, fn_score) in zip(true_positives, false_negatives):
                tp_rate_list.append(tp_score / (tp_score + fn_score))
            return tp_rate_list

        true_positive_rate = true_positive_computation(self.true_positives, self.false_negatives)
        val_true_positive_rate = true_positive_computation(self.val_true_positives, self.val_false_negatives)

        tpr_plots = [go.Scatter(x=self.epoch_list,
                                y=true_positive_rate,
                                mode='lines',
                                name='True Positive Rate',
                                line=dict(width=4)),
                     go.Scatter(x=self.epoch_list,
                                y=val_true_positive_rate,
                                mode='lines',
                                name='Validation True Positive Rate',
                                line=dict(width=4))]

        self.tpr_figure = go.Figure(data=tpr_plots)
        self.subplot_name_list.append('True Positive Rate Graph')
        self.figure_x_axes_list.append("Epochs")
        self.figure_y_axes_list.append("True Positive Rate")
        self.subplot_list.append(self.tpr_figure)

    def true_negative_graph(self):

        def true_negative_computation(true_negatives, false_positives):
            tn_rate_list = []
            for (tn_score, fp_score) in zip(true_negatives, false_positives):
                tn_rate_list.append(tn_score / (tn_score + fp_score))
            return tn_rate_list

        true_negative_rate = true_negative_computation(self.true_negatives, self.false_positives)
        val_true_negative_rate = true_negative_computation(self.val_true_negatives, self.val_false_positives)

        tnr_plots = [go.Scatter(x=self.epoch_list,
                                y=true_negative_rate,
                                mode='lines',
                                name='True Negative Rate',
                                line=dict(width=4)),
                     go.Scatter(x=self.epoch_list,
                                y=val_true_negative_rate,
                                mode='lines',
                                name='Validation True Negative Rate',
                                line=dict(width=4))]

        self.tnr_figure = go.Figure(data=tnr_plots)
        self.subplot_name_list.append('True Negative Rate Graph')
        self.figure_x_axes_list.append("Epochs")
        self.figure_y_axes_list.append("True Negative Rate")
        self.subplot_list.append(self.tnr_figure)

    def subplot_creation(self, row_size, col_size):

        def row_column_index_creator(index_row_size, index_col_size):  # Outputs every possible row/col index pair
            row_col_index_list_func = []
            index_row_size += 1
            index_col_size += 1
            [row_col_index_list_func.append([row, column]) for row in range(1, index_row_size) for column in
             range(1, index_col_size)]
            return row_col_index_list_func

        def axes_title_creator(x_axes_list, y_axes_list):  # Creates list of x and y axes titles
            x_y_axes_title_list = []
            for (x_axes_title, y_axes_title) in zip(x_axes_list, y_axes_list):
                x_y_axes_title_list.append([x_axes_title, y_axes_title])
            return x_y_axes_title_list

        row_col_index_list = row_column_index_creator(row_size, col_size)
        x_y_axes_list = axes_title_creator(self.figure_x_axes_list, self.figure_y_axes_list)
        metric_subplot = make_subplots(rows=row_size, cols=col_size, subplot_titles=self.subplot_name_list)

        for plot, row_col, x_y_ax in zip(self.subplot_list, row_col_index_list, x_y_axes_list):
            x_axes = x_y_ax[0]
            y_axes = x_y_ax[1]
            row_index = row_col[0]
            col_index = row_col[1]

            metric_subplot.update_xaxes(title_text=x_axes, row=row_index, col=col_index)
            metric_subplot.update_yaxes(title_text=y_axes, row=row_index, col=col_index)

            for trace in plot.data:
                metric_subplot.append_trace(trace, row=row_index, col=col_index)

        offline.plot(metric_subplot,
                     filename=f'{self.metric_dir}_training_metrics.html',
                     auto_open=False)

    def confusion_matrix(self, class_indices):
        train_z = [[self.last_true_negatives, self.last_false_negatives],
                   [self.last_true_positives, self.last_false_positives]]

        val_z = [[self.last_val_true_negatives, self.last_val_false_negatives],
                 [self.last_val_true_positives, self.last_val_false_positives]]

        x = ['True', 'False']
        y = ['Negative (0)', 'Positive (1)']

        # Turn each item in z into a string for annotation only
        def string_annotation_converter(z_data):
            z_text = [[str(y_data) for y_data in x_data] for x_data in z_data]
            return z_text

        # set up figure
        train_confusion_mat = ff.create_annotated_heatmap(train_z, x=x, y=y,
                                                          colorscale='Viridis',
                                                          annotation_text=string_annotation_converter(train_z))
        train_confusion_mat.add_annotation(text=f'Training CM - {str(class_indices)}',
                                           align='left',
                                           showarrow=False,
                                           xref='paper',
                                           yref='paper',
                                           x=0.5,
                                           y=1.1,
                                           bordercolor='black',
                                           borderwidth=1)

        val_confusion_mat = ff.create_annotated_heatmap(val_z, x=x, y=y,
                                                        colorscale='Viridis',
                                                        annotation_text=string_annotation_converter(val_z))
        val_confusion_mat.add_annotation(text=f'Validation CM - {str(class_indices)}',
                                         align='left',
                                         showarrow=False,
                                         xref='paper',
                                         yref='paper',
                                         x=0.5,
                                         y=1.1,
                                         bordercolor='black',
                                         borderwidth=1)

        # Takes figures and appends them to single html file
        def figures_to_html(figs, filename=f'{self.metric_dir}_training_confusion_matrix.html'):
            dashboard = open(filename, 'w')
            dashboard.write("<html><head></head><body>" + "\n")
            for fig in figs:
                inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
                dashboard.write(inner_html)
            dashboard.write("</body></html>" + "\n")

        figures_to_html([train_confusion_mat, val_confusion_mat])


class EvaluationDataVisualization:
    def __init__(self, metric_data, metric_dir):
        self.metric_dir = metric_dir
        self.metric_data = metric_data

        self.epoch = metric_data['epoch']
        self.loss = metric_data['loss']
        self.accuracy = metric_data['accuracy']
        self.auc = metric_data['auc']
        self.recall = metric_data['recall']
        self.precision = metric_data['precision']
        self.false_positive = metric_data['false_positives_3']
        self.true_negative = metric_data['true_negatives_3']
        self.false_negative = metric_data['false_negatives_3']
        self.true_positive = metric_data['true_positives_3']

    def eval_barchart(self):

        def list_to_average(metric):  # Returns average from a list of ints
            metric_list = []
            for num in metric:
                metric_list.append(num)
            return sum(metric_list) / len(metric_list)

        def metrics_barchart():
            x_labels = ['Loss', 'Accuracy', 'AUC', 'Recall', 'Precision']

            y_metric_list = [list_to_average(self.loss), list_to_average(self.accuracy), list_to_average(self.auc),
                             list_to_average(self.recall), list_to_average(self.precision)]

            color = ['pink', 'gold', 'springgreen', 'rgb(29, 105, 150)', 'rgb(228, 26, 28)']

            evaluation_barchart = go.Figure(go.Bar(x=x_labels, y=y_metric_list, marker=dict(color=color)))

            evaluation_barchart.add_annotation(text=f'Evaluation Bar Charts',
                                               align='left',
                                               showarrow=False,
                                               xref='paper',
                                               yref='paper',
                                               x=0.5,
                                               y=1.1,
                                               bordercolor='black',
                                               borderwidth=1)
            return evaluation_barchart

        def boolean_metrics_barchart():
            x_labels = ['False Positive', 'True Negative',
                        'False Negative', 'True Positive']

            y_metric_list = [list_to_average(self.false_positive), list_to_average(self.true_negative),
                             list_to_average(self.false_negative), list_to_average(self.true_positive)]

            color = ['red', '#00D', 'red', '#00D']

            bool_evaluation_barchart = go.Figure(go.Bar(x=x_labels, y=y_metric_list, marker=dict(color=color)))
            return bool_evaluation_barchart

        def figures_to_html(figs, filename=f'{self.metric_dir}_evaluation_barchart.html'):
            dashboard = open(filename, 'w')
            dashboard.write("<html><head></head><body>" + "\n")
            for fig in figs:
                inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
                dashboard.write(inner_html)
            dashboard.write("</body></html>" + "\n")

        figures_to_html([metrics_barchart(), boolean_metrics_barchart()])
