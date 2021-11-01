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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    numerical_features = x_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = x_train.select_dtypes(include=['object']).columns

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

    x_train = preprocessor_pipeline.fit(x_train)
    x_test = preprocessor_pipeline.fit(x_test)

    return x_train, x_test, y_train, y_test
