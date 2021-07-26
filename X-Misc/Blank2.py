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


def row_column_index_creator(index_row_size, index_col_size):
    row_col_index_list = []
    index_row_size -= 1
    index_col_size += 1
    for row_index in range(index_col_size):
        row_index += 1
        for col_index in range(index_row_size):
            col_index += 1
            row_col_index_list.append([row_index, col_index])
    return row_col_index_list


print(row_column_index_creator(6, 8))
