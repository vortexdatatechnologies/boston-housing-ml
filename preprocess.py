from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Load the Boston Housing dataset
def load_data():
    return fetch_openml(name="boston", version=1, as_frame=True).frame

# Splitting the data
def split_data(data, test_size=0.2, random_state=42):
    return train_test_split(data, test_size=test_size, random_state=random_state)

# Handling missing values
def handle_missing_values(train_set, test_set, numerical_cols):
    for df in [train_set, test_set]:
        for col in numerical_cols:
            mean_val = train_set[col].mean()
            df[col] = df[col].fillna(mean_val)
    return train_set, test_set

# Removing duplicates
def remove_duplicates(train_set):
    duplicates = train_set.duplicated()
    if duplicates.sum() > 0:
        train_set = train_set.loc[~duplicates]
    return train_set

# Feature Engineering
def feature_engineering(train_set, test_set):
    for df in [train_set, test_set]:
        df['LSTAT_squared'] = df['LSTAT'] ** 2
        df['CRIM_log'] = np.log(df['CRIM'])
    return train_set, test_set

# Separating the features and labels
def separate_features_labels(train_set, test_set):
    X_train = train_set.drop('MEDV', axis=1)
    y_train = train_set['MEDV']
    X_test = test_set.drop('MEDV', axis=1)
    y_test = test_set['MEDV']
    return X_train, X_test, y_train, y_test

# Scaling the features
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    # Save the scaler object for future use
    joblib.dump(scaler, 'scaler.pkl')
    return X_train_scaled, X_test_scaled

def preprocess_data():
    boston = load_data()
    train_set, test_set = split_data(boston)
    numerical_cols = train_set.select_dtypes(include=['float64', 'int64']).columns
    train_set, test_set = handle_missing_values(train_set, test_set, numerical_cols)
    train_set = remove_duplicates(train_set)
    train_set, test_set = feature_engineering(train_set, test_set)
    X_train, X_test, y_train, y_test = separate_features_labels(train_set, test_set)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data()
    print(X_train_scaled.head())
