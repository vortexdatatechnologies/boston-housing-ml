from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the Boston Housing dataset
def preprocess_data():
    boston_dataset = fetch_openml(name="boston", version=1, as_frame=True)
    boston = boston_dataset.frame

    # Split the data into training set and test set
    train_set, test_set = train_test_split(boston, test_size=0.2, random_state=42)

    # Data cleaning: fill NA values with the column mean
    train_set.fillna(train_set.mean(), inplace=True)
    test_set.fillna(train_set.mean(), inplace=True)  # use mean from training set

    # Check for duplicate rows in the training set
    duplicates = train_set.duplicated()
    print(f"Number of duplicate rows = {duplicates.sum()}")

    # Remove duplicates if any
    if duplicates.sum() > 0:
        train_set = train_set[~duplicates]

    # Feature Engineering: Add some new features (for example, square of LSTAT and log of CRIM)
    train_set['LSTAT_squared'] = train_set['LSTAT'] ** 2
    train_set['CRIM_log'] = np.log(train_set['CRIM'])

    # Apply same transformations to the test set
    test_set['LSTAT_squared'] = test_set['LSTAT'] ** 2
    test_set['CRIM_log'] = np.log(test_set['CRIM'])

    # Separate the features from the labels
    train_set_features = train_set.drop('MEDV', axis=1)
    train_set_labels = train_set['MEDV']

    test_set_features = test_set.drop('MEDV', axis=1)
    test_set_labels = test_set['MEDV']

    # Feature Scaling: Standardize the features
    scaler = StandardScaler()
    train_set_scaled = pd.DataFrame(scaler.fit_transform(train_set_features), columns=train_set_features.columns)
    test_set_scaled = pd.DataFrame(scaler.transform(test_set_features), columns=test_set_features.columns)
    return train_set_scaled, test_set_scaled, train_set_labels, test_set_labels

# Now your data is ready for machine learning modeling!
if __name__ == "__main__":
    train_set_scaled, test_set_scaled, train_set_labels, test_set_labels = preprocess_data()
    print(train_set_scaled.head())
