# Data Preprocessing

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from icecream import ic


def diamond_preprocess(data_dir):
    data = pd.read_csv(data_dir)
    cleaned_data = data.drop(['id', 'depth_percent'], axis=1)  # Features I don't want

    x = cleaned_data.drop(['price'], axis=1)  # Train data
    y = cleaned_data['price']  # Label data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99)

    numerical_features = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = x_train.select_dtypes(include=['object']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Fill in missing data with median
        ('scaler', StandardScaler())  # Scale data
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Fill in missing data with 'missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One hot encode categorical data
    ])

    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit to the training data
    preprocessor_pipeline.fit(x_train)
    ic(x_train.shape)
    ic(y_train.head(5))
    # preprocessor_pipeline.fit(y_train)

    # Apply the pipeline to the training and test data
    x_train_pipe = preprocessor_pipeline.transform(x_train)
    x_test_pipe = preprocessor_pipeline.transform(x_test)
    # y_train_pipe = preprocessor_pipeline.transform(y_train)
    # y_test_pipe = preprocessor_pipeline.transform(y_test)

    x_train = pd.DataFrame(data=x_train_pipe)
    x_test = pd.DataFrame(data=x_test_pipe)
    # y_train = pd.DataFrame(data=y_train_pipe)
    # y_test = pd.DataFrame(data=y_test_pipe)

    ic(x_train)
    ic(type(x_train))
    ic(x_test)
    ic(type(x_test))
    ic(y_train)
    ic(type(y_train))
    ic(y_test)
    ic(type(y_test))

    return x_train, x_test, y_train, y_test
