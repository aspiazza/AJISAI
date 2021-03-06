# Data Preprocessing

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def diamond_preprocess(data_dir):
    data = pd.read_csv(data_dir)
    cleaned_data = data.drop(['id', 'depth_percent'], axis=1)  # Features I don't want

    x = cleaned_data.drop(['price'], axis=1)  # Train data
    y = cleaned_data['price'].copy()  # Label data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99)

    # Reshape for single feature df
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

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
    target_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor_pipeline = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Fit to the training data
    preprocessor_pipeline.fit(x_train)
    target_pipeline.fit(y_train)

    # Apply the pipeline to the training and test data
    pp_x_train = pd.DataFrame(preprocessor_pipeline.transform(x_train))
    pp_x_test = pd.DataFrame(preprocessor_pipeline.transform(x_test))
    pp_y_train = pd.DataFrame(target_pipeline.transform(y_train))
    pp_y_test = pd.DataFrame(target_pipeline.transform(y_test))

    return pp_x_train, pp_x_test, pp_y_train, pp_y_test


def feat_removal_diamond_preprocess(data_dir):
    data = pd.read_csv(data_dir)
    cleaned_data = data.drop(['id', 'depth_percent', 'length'], axis=1)  # Features I don't want

    x = cleaned_data.drop(['price'], axis=1)  # Train data
    y = cleaned_data['price'].copy()  # Label data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99)

    # Reshape for single feature df
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

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
    target_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor_pipeline = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Fit to the training data
    preprocessor_pipeline.fit(x_train)
    target_pipeline.fit(y_train)

    # Apply the pipeline to the training and test data
    pp_x_train = pd.DataFrame(preprocessor_pipeline.transform(x_train))
    pp_x_test = pd.DataFrame(preprocessor_pipeline.transform(x_test))
    pp_y_train = pd.DataFrame(target_pipeline.transform(y_train))
    pp_y_test = pd.DataFrame(target_pipeline.transform(y_test))

    return pp_x_train, pp_x_test, pp_y_train, pp_y_test


def strat_diamond_preprocess(data_dir):
    data = pd.read_csv(data_dir)
    cleaned_data = data.drop(['id', 'depth_percent'], axis=1)  # Features I don't want

    x = cleaned_data.drop(['price'], axis=1)  # Train data
    y = cleaned_data['price'].copy()  # Label data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99)

    # Reshape for single feature df
    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

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
    target_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor_pipeline = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Fit to the training data
    preprocessor_pipeline.fit(x_train)
    target_pipeline.fit(y_train)

    # Apply the pipeline to the training and test data
    pp_x_train = pd.DataFrame(preprocessor_pipeline.transform(x_train))
    pp_x_test = pd.DataFrame(preprocessor_pipeline.transform(x_test))
    pp_y_train = pd.DataFrame(target_pipeline.transform(y_train))
    pp_y_test = pd.DataFrame(target_pipeline.transform(y_test))

    return pp_x_train, pp_x_test, pp_y_train, pp_y_test
