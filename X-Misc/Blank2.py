'''self.loss_graph.update_layout(
    font_color='black',
    title_font_color='black',
    title=dict(text='Loss Graph',
               font_size=30),
    xaxis_title=dict(text='Epochs',
                     font_size=25),
    yaxis_title=dict(text='Loss',
                     font_size=25),
    legend=dict(font_size=15))

self.error_rate_figure.update_layout(
    font_color='black',
    title_font_color='black',
    title=dict(text='Error Rate Graph',
               font_size=30),
    xaxis_title=dict(text='Epochs',
                     font_size=25),
    yaxis_title=dict(text='Error Rate',
                     font_size=25),
    legend=dict(font_size=15))

self.recall_figure.update_layout(
    font_color='black',
    title_font_color='black',
    title=dict(text='Recall Graph',
               font_size=30),
    xaxis_title=dict(text='Epochs',
                     font_size=25),
    yaxis_title=dict(text='Recall',
                     font_size=25),
    legend=dict(font_size=15))

self.precision_figure.update_layout(
    font_color='black',
    title_font_color='black',
    title=dict(text='Precision Graph',
               font_size=30),
    xaxis_title=dict(text='Epochs',
                     font_size=25),
    yaxis_title=dict(text='Precision',
                     font_size=25),
    legend=dict(font_size=15))

self.f1_figure.update_layout(
    font_color='black',
    title_font_color='black',
    title=dict(text='F1 Graph',
               font_size=30),
    xaxis_title=dict(text='Epochs',
                     font_size=25),
    yaxis_title=dict(text='F1 Score',
                     font_size=25),
    legend=dict(font_size=15))
'''

empty_list = []
row = [2, 4, 6, 8, 10]
col = [1, 3, 5, 7, 9]

for (r, c) in zip(row, col):
    empty_list.append([r, c])

for pair in empty_list:
    print(pair[0])
    print(pair[1])
    print('------')
