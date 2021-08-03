# AJISAI NOTES

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