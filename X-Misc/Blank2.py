# Nucleus example code

# Nucleus code
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Model class
class catdogModel:  # Include logging and data viz throughout
    def __init__(self, model_name, datafile):
        self.datafile = pd.read_csv(datafile)
        self.model_name = model_name

    def preprocess(self, test_size):
        X = np.array(self.datafile[['Humidity', 'Pressure (millibars)']])
        y = np.array(self.datafile['Temperature (C)'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                random_state=42)
        # Plot data exploration metrics

    def baseline(self, pp_data):
        RNN.train(pp_data)
        # Plot results in subplot
        Tree.train(pp_data)
        # Plot results in subplot

    def fit(self):
        self.model = self.random_forest.fit(self.X_train, self.y_train)
        # plotting fit data
        # Saving model

    def predict(self, input_value):
        if input_value == None:
            result = self.random_forest.fit(self.X_test)
        else:
            result = self.random_forest.fit(np.array([input_value]))
        return result
        # Plotting prediction data

    def saving(self):
        log_dir = 'Model-Graphs&Logs\\' + self.model_name + '\\Logs'
        metric_dir = 'Model-Graphs&Logs\\' + self.model_name + '\\Metric-Graphs'
        if not os.path.isfile(log_dir):
            os.mkdir(log_dir)

        if not os.path.isfile(metric_dir):
            os.mkdir(metric_dir)
        return None


# Executor
if __name__ == '__main__':
    model_instance = catdogModel("CatDog_Model", "data.csv")
    model_instance.preprocess(0.2)
    model_instance.fit()
    print(model_instance.predict(1))
    print("Accuracy: ", model_instance.model.score(model_instance.X_test, model_instance.y_test))


# print out requirements.txt
'''
pipreqs AJISAI-Project\Web-App\
'''

# Useful stuff to know
'''
- Guides
    + Fast-API Templates:
        > https://www.youtube.com/watch?v=JC5q22g3yQM

    + ML Model on Fast-API
        > https://www.youtube.com/watch?v=Mw9etoRz0Ic

    + Docker and Fast-API
        > https://fastapi.tiangolo.com/deployment/docker/
        > https://towardsdatascience.com/tensorflow-model-deployment-using-fastapi-docker-4b398251af75
        > https://medium.com/swlh/python-with-docker-fastapi-c4c304c7a93b
        > https://www.youtube.com/watch?v=2a5414BsYqw
        
    + Metric information
        > https://neptune.ai/blog/evaluation-metrics-binary-classification
'''

# Code
'''
. MinMaxScaler (sklearn.preprocessing) = Normalizing binary data to 1 or 0
. OnehotEncoder (sklearn.preprocessing) = One hot encode categorical data
. random.sample(glob.blob('cat*'), 100) = Grabs 100 files containing word cat
. shutil.move(source, dest) = Moves files
. random.choice(os.listdir('example/directory')) = choose random file/image
. StratifiedShuffleSplit(n_splts=, test_size=) = When you want your data stratified
. SimpleImputer(strategy= "median") = Handling missing data


.vectorizer = CountVectorizer()
 vectorizer.fit_transform(train_x)
 vectorizer.transform(test_x)       #To vectorized words


. #Normalize data with tensorflow (Categories and numeric)
 CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                   'embark_town', 'alone']
   NUMERIC_COLUMNS = ['age', 'fare']
   feature_columns = []

   for feature_name in CATEGORICAL_COLUMNS:
        # gets a list of all unique values from given feature column
        vocabulary = dftrain[feature_name].unique()
        #Maps categorey with unique values
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

   for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
'''

# Plotting average image
'''import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# making n X m matrix
def img2np(path, list_of_filename, size = (64, 64)):
    # iterating through each file
    for fn in list_of_filename:
        fp = path + fn
        current_image = image.load_img(fp, target_size = size,
                                       color_mode = 'grayscale')
        # covert image to a matrix
        img_ts = image.img_to_array(current_image)
        # turn that into a vector / 1D array
        img_ts = [img_ts.ravel()]
        try:
            # concatenate different images
            full_mat = np.concatenate((full_mat, img_ts))
        except UnboundLocalError:
            # if not assigned yet, assign one
            full_mat = img_ts
    return full_mat

# run it on our folders
normal_images = img2np(f'{train_dir}/NORMAL/', normal_imgs)
pnemonia_images = img2np(f'{train_dir}/PNEUMONIA/', pneumo_imgs)

def find_mean_img(full_mat, title, size = (64, 64)):
    # calculate the average
    mean_img = np.mean(full_mat, axis = 0)
    # reshape it back to a matrix
    mean_img = mean_img.reshape(size)
    plt.imshow(mean_img, vmin=0, vmax=255, cmap='Greys_r')
    plt.title(f'Average {title}')
    plt.axis('off')
    plt.show()
    return mean_img

norm_mean = find_mean_img(normal_images, 'NORMAL')
pneu_mean = find_mean_img(pnemonia_images, 'PNEUMONIA')'''