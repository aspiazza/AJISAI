# Data Visualization
from plotly.subplots import make_subplots
import plotly.graph_objects as go

epoch_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loss_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
val_loss_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
error_rate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
val_error_rate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

layout_list = []

loss_plots = [go.Scatter(x=epoch_list,
                         y=loss_list,
                         mode='lines',
                         name='Loss',
                         line=dict(width=4)),
              go.Scatter(x=epoch_list,
                         y=val_loss_list,
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

loss_figure = go.Figure(data=loss_plots)
layout_list.append(loss_layout)

error_plots = [go.Scatter(x=epoch_list,
                          y=loss_list,
                          mode='lines',
                          name='Error Rate',
                          line=dict(width=4)),
               go.Scatter(x=epoch_list,
                          y=val_loss_list,
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

error_figure = go.Figure(data=error_plots)
layout_list.append(error_rate_layout)

metric_figure = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{}, {}],
           [{}, {}]])

for t in loss_figure.data:
    metric_figure.append_trace(t, row=1, col=1)
for t in error_figure.data:
    metric_figure.append_trace(t, row=1, col=2)

for (figure, layout) in zip(metric_figure, layout_list):
    figure.update_layout(layout)

metric_figure.show()
