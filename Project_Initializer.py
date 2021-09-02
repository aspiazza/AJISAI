# Project Initializer
import os


def make_project(project_name):  # Function to make files and directories
    if not os.path.isdir(f'Model-Graphs&Logs\\Model-Data_{project_name}\\Logs'):
        os.makedirs(f'Model-Graphs&Logs\\Model-Data_{project_name}\\Logs')
    else:
        pass

    if not os.path.isdir(f'Model-Graphs&Logs\\Model-Data_{project_name}\\Metric-Graphs'):
        os.makedirs(f'Model-Graphs&Logs\\Model-Data_{project_name}\\Metric-Graphs')
    else:
        pass

    file_list = [f'Extraction-Scripts\\Extract_{project_name}.py',
                 f'Data-Exploration\\Explore_{project_name}.py',
                 f'Nucleus_{project_name}.py',
                 f'Pipeline\\Callbacks\\Callbacks_{project_name}.py',
                 f'Pipeline\\Data_Visual\\Data_Visual_{project_name}.py',
                 f'Pipeline\\Grid_Search\\Grid_Search_{project_name}.py',
                 f'Pipeline\\Models\\Model_{project_name}.py',
                 f'Pipeline\\Prediction\\Prediction_{project_name}.py',
                 f'Pipeline\\Preprocess\\Preprocess_{project_name}.py']

    for file in file_list:
        if not os.path.isfile(file):
            open(file, mode='a').close()
        else:
            continue


# TODO: Update as Nucleus code progresses
def make_webapp(project_name):  # Function to make Web-App files and directories
    web_app_dir = f'Web-Apps\\Web-App_{project_name}'
    os.mkdir(web_app_dir)

    file_list = [f'{web_app_dir}\\API-{project_name}.py',
                 f'{web_app_dir}\\Dockerfile-{project_name}.dockerfile',
                 f'{web_app_dir}\\Predictor-{project_name}.py',
                 f'{web_app_dir}\\WebPage-{project_name}.html',
                 f'{web_app_dir}\\style-{project_name}.css']

    for file in file_list:
        if not os.path.isfile(file):
            open(file, mode='a').close()
        else:
            continue


def populate_nucleus(project_name):
    with open(f'Nucleus_{project_name}.py', 'a') as nucleus_file:
        nucleus_file.write('''
class placeholder:
        def __init__(self, model_name, version, datafile, saved_weights):
        self.datafile = datafile
        self.version_model_name = f'{version}_{model_name}'

        self.saved_weights = f'{saved_weights}\\{self.version_model_name}'
        self.log_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Logs\\{self.version_model_name}'
        self.metric_dir = f'Model-Graphs&Logs\\Model-Data_{model_name}\\Metric-Graphs\\{self.version_model_name}'

    def preprocess(self):
        pass

    def model(self):
        pass

    def grid_search(self):
        pass

    def training(self, callback_bool):
        pass

    def graphing(self, csv_file):
        pass

    def evaluate(self, saved_weights_dir, callback_bool):
        pass

    def evaluate_graphing(self, csv_file):
        pass

    @staticmethod
    def predict(saved_weights_dir, prediction_data):
        pass


# Executor
if __name__ == '__main__':
    model_instance = placeholder(model_name=placeholder, version=placeholder, datafile=placeholder, saved_weights)
    # model_instance.preprocess()
    # model_instance.model()
    # model_instance.grid_search()
    # model_instance.training(callback_bool=True)
    # model_instance.graphing(csv_file=None)
    # model_instance.evaluate(saved_weights_dir=None, callback_bool=True)
    # model_instance.evaluate_graphing(csv_file=None)
    # model_instance.model_predict(saved_weights_dir=None ,prediction_data=None)
''')


project_name_input = input('Please enter project name: ')
webapp_bool = input('Would you like a webapp dir (y/n): ')

if webapp_bool.lower() == 'y':  # Bool if you want Web files
    make_project(project_name_input)
    populate_nucleus(project_name_input)
    print('\nProject files successfully created')
    make_webapp(project_name_input)
    print('Web-App files successfully created')
else:
    make_project(project_name_input)
    populate_nucleus(project_name_input)
    print('Project files successfully created')
