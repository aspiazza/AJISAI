# AJISAI Project README

### Purpose:

The purpose of this project was to create an end-to-end machine learning web application. This was created to gain a
deeper understanding of Python, Machine Learning, Web Scraping, Plotly Graphing, API Developing, Docker, Cloud
Environments, HTML/CSS and production worthy code. It was also to prove that I had the skills and drive to accomplish
such a large and complex project. I plan on using this project as a long-term medium that I continually update and use
to complete other ML projects in the future. Because of this, I expect there to be many improvements to come in the
following months/years and additions. These improvements would consist of organizing folder structure, cleaning code,
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
        + Model-Data_{model name}
            + Logs
            + Metric-Graphs
    + Pipeline
        + Baseline_Models
        + Data_Visual
        + Models
        + Preprocess
    + Web-Apps
        + Web-App_{model name}
    + Nucleus_{model name}.py
    + Project_Initializer.py
    + README.md
- /F:/Data-Warehouse
- /F:/Saved-Models

### What The Directories Contain:

- Data-Exploration
    + glob, numpy, os, pandas, PIL, plotly, random
      > - Script that extract useful information and output it to visualization directory
      > - Used to explore data, stats, correlation coefficients, clustering, etc

- Extraction-Scripts
    + beautifulsoup, glob, os, random, requests, selenium, shutil
      > - Script that extracts data and organizes it into /F:/Data-Warehouse
      > - Extraction scripts will be unique and not in the nucleus script as data extraction will always be unique depending on task.
      > - All data will output to a CSV, Tensorflow Dataframe, json, or images

- Model-Graphs&Logs
    + None
      > - Stores data exploration, metrics, logs, json history, and graphs
      > - Will be images, json, html, and text files
      > - Will come from preprocessing, baselines, training, exploration, and prediction processes
      > - Model subdirectory used for organization

- Pipeline\Baseline-Models
    + Keras, SciKitLearn, Tensorflow, numba, numpy, pandas
      > - Will contain basic models used to get baseline of data
      > - Trees, neural nets, machine learning, etc
      > - Would like to pass data into one or multiple models asynchronously?

- Pipeline\Data-Visual
    + Plotly, SciKitLearn
      > - Visualization component that is itself a class
      > - Data Viz will be broken up into different functions within a single custom library
      > - Will output training, baseline, post-preprocessed, and prediction subplots
      > - Take in argument of model name which will determine directory and filename

- Pipeline\Models
    + Keras, SciKitLearn, Tensorflow
      > - Will contain the actual models and there parameters
      > - Training and grid tuning of model, testing, and displaying plot model and summary
      > - Models will be made using functional API and sequential
      > - Results of training should be outputted to Data Visual directory and logs
      > - Implement Freezing, Pretrained layers, other techniques later on

- Pipeline\Preprocess
    + Keras, SciKitLearn, Tensorflow, numba, numpy, os, pandas
      > - Data preprocessing component that will be in a class but not always
      > - Will be working in the Data Preprocessing directory
      > - Data PP will be broken up into different functions within a single custom library
      > - Generators and Feature engineering take place here as well with stats outputted to graphs

- Web-Apps
    + AWS, Azure, Docker, Fast-API, css, html, jinja, pipreq, plotly, requests, templates
      > - Contains directories for each model web app
      > - Create an application with html, python, and css
      > - Allow user to input data, and model outputs prediction
      > - Put application in docker container
      > - Will need to create docker file to set up container
      > - Will need to create a requirements.txt to download packages
      > - Maybe mount this on a cloud service

- Nucleus.py
    + custom libraries, datetime, keras, numba, os, sys, tensorflow
      > - A nucleus file executes all functions related to training/testing
      > - Will be a complex class that does all the above steps
      > - Will call functions to preprocess, train, visualize, tune, and save model/data
      > - Implements callbacks

- Project_Initializer.py
    + os
      > - Creates all files and directories needed when starting a new project
      > - Argument that is passed in is name ex. 'dog-cat'

- Data-Warehouse
    + None
      > - Contains data used in projects

- Saved-Models
    + None
      > - Will contain saved model weights

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
