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

from icecream import ic

'''
class TestClass:
    def __init__(self):
        self.variable = 1
        self.string = "Andrew"

    def printer(self):
        print("Executed printer()")

        class AndrewPrinter:
            print("Executed andrewPrinter")

            def function_printer(self, outer_class):
                print(f'Executed function_printer')
                print(outer_class.self.string)

        ap = AndrewPrinter().function_printer(TestClass)
        return ap


TestClass().printer()'''


class Outer:
    def __init__(self, file_list):
        self.file_list = file_list
        self.inner = self.Inner(self)

    class Inner:
        def __init__(self, outer_parent):
            print(outer_parent.file_list)
            pass
