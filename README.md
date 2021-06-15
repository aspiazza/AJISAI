# AJISAI Project README

### Purpose:

The purpose of this project was to create an end-to-end machine learning web application. This was created to not only
improve my skills as a machine learning practitioner, python developer, HTML/CSS developer, and Docker user but to prove
that I had the skills and drive to get it done. I plan on using this project as a long-term medium I continually update
and use to complete other ML-related projects in the future. Because of this, I expect there to be many improvements to
come in the following months/years. These improvements would consist of organizing folder structure, cleaning code,
adding new models, updating documentation, and automating tasks if need be. Below are some skills I utilized in this
project:

- Python Code
- Web scraping and Data Extraction
- Data Visualization
- Backend Fast-API Programming
- Docker Containers
- Cloud Environment
- HTML/CSS Web App Development
- Machine Learning Techniques and Models
- Clean, Reusable, Production Worthy Code

### Directory Structure:

- AJISAI-Project
    + Data-Exploration
    + Extraction-Scripts
    + Model-Graphs&Logs
        - Model-Data_{model name}
            + Logs
            + Metric-Graphs
    + Pipeline
        - Baseline-Models
        - Data-Visual
        - Models
        - Preprocess
    + Web-Apps
        - Web-App_{model name}
    + Nucleus_{model name}.py
    + Project_Initializer.py
    + README.md
- /F:/Data-Warehouse
- /F:/Saved-Models

### What The Directories Contain:

- Data-Exploration
    + plotly, numpy, random, pandas, PIL, os, glob
      > - Script that extract useful information and output it to visualization directory
      > - Used to explore data, stats, correlation coefficients, clustering, etc

- Extraction-Scripts
    + requests, beautifulsoup, selenium, os, glob, shutil, random
      > - Script that extracts data and organizes it into /F:/Data-Warehouse
      > - Extraction scripts will be unique and not in the nucleus script as data extraction will always be unique depending on task.
      > - All data will output to a CSV, Tensorflow Dataframe, or images

- Model-Graphs&Logs
    + None
      > - Stores data exploration, metrics, logs, json history, and graphs
      > - Will be images, json, and text files
      > - Will come from preprocessing, baselines, training, and other processes
      > - Model subdirectory will just be used for organization

- Nucleus.py
    + os, datetime, custom libraries
      > - A nucleus file executes all functions related to training/testing
      > - Will be a complex class that does all the above steps
      > - Will call functions to preprocess, train, visualize, tune, and save model/data
      > - Implement logging

- Pipeline\Baseline-Models
    + SciKitLearn, Keras, Tensorflow, numpy, pandas, numba
      > - Will contain basic models used to get baseline of data
      > - Trees, neural nets, machine learning, etc
      > - Would like to pass data into one or multiple models asynchronously?

- Pipeline\Data-Visual
    + Plotly, os, matplotlib, SciKitLearn
      > - Visualization component that can be modular and change on the fly
      > - Data Viz will be broken up into different functions within a single custom library
      > - Will output exploration, training, and baseline subplots
      > - Take in argument of model name which will determine directory and filename

- Pipeline\Models
    + os, SciKitLearn, Keras, Tensorflow, numpy, Plotly, numba
      > - Training and grid tuning of model, testing, and displaying metrics
      > - Will contain the actual models and there parameters
      > - Code will contain functions that will train, validate, and test model
      > - Models will be made using functional API and sequential
      > - Results of training should be outputted to Data Visual directory
      > - Take in boolean argument of save model
      > - Implement Callbacks, Freezing, Pretrained layers (Maybe later on)

- Pipeline\Preprocess
    + SciKitLearn, Keras, Tensorflow, numpy, pandas, numba
      > - Data preprocessing component that can be modular and change on the fly
      > - Will be working in the Data Preprocessing directory
      > - Data PP will be broken up into different functions within a single custom library
      > - Not sure if each preprocessing step will be unique to an ML model/problem
      > - Generator? Feature engineering?

- Web-Apps
    + Docker, Azure, AWS, pipreq, Fast-API, html, css, requests, plotly, jinja, templates
      > - Create an application with html, python, and css
      > - Contains directories for each model web app
      > - Allow user to input data, and model outputs prediction
      > - Put application in docker container
      > - Will need to create docker file to set up container
      > - Will need to create a requirements.txt to download packages
      > - Maybe mount this on a cloud service

- Project_Initializer.py
    + os
      > - Creates all files and directories needed when starting a new project
      > - Argument that is passed in is name ex. 'dog-cat'

- Data-Warehouse
    + None
      > - Contains data used in training models and visualization

- Saved-Models
    + SciKitLearn, Keras, Tensorflow, numpy, Plotly, numba
      > - Will contain model weights and nucleus.py files from previous projects
      
### Logic Steps:

> 1. Use web scraping libraries that extract data online and saves to the Data-Warehouse directory
> 2. Explore data and use visualization libraries to show metadata
> 3. All within the Nucleus file ->
> 4. Preprocess data using different techniques and methods
> 5. Get a baseline of the data and output visualization
> 6. Create and train model using all available techniques and methods
> 7. Output logs and visualization of model training and testing results
> 8. Save model weights and Nucleus file.
> 9. Create a html/css webpage to implement model in
> 10. Create a backend api with Fast-API to be able to take in and serve model results
> 11. Package Web app into Docker container using Docker file
> 12. Place container on cloud platform
