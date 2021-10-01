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

    if not os.path.isdir(f'Web-Apps\\Web-App_{project_name}'):
        os.makedirs(f'Web-Apps\\Web-App_{project_name}')
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


project_name_input = input('Please enter project name: ')
make_project(project_name_input)
print('Project files successfully created')
