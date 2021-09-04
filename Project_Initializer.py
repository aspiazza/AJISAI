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


project_name_input = input('Please enter project name: ')
webapp_bool = input('Would you like a webapp dir (y/n): ')

if webapp_bool.lower() == 'y':  # Bool if you want Web files
    make_project(project_name_input)
    print('\nProject files successfully created')
    make_webapp(project_name_input)
    print('Web-App files successfully created')
else:
    make_project(project_name_input)
    print('Project files successfully created')
