import plotly.figure_factory as ff
import plotly.graph_objects as go
from icecream import ic


def confusion_matrix():
    z = [[11, 12],
         [23, 24]]

    x = ['Positive', 'Negative']
    y = ['Negative', 'Positive']

    # Turn each item in z into a string for annotation only
    z_text = [[str(y) for y in x] for x in z]

    # set up figure
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>')

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Actual Value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Predicted Value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    fig.show()

confusion_matrix()
