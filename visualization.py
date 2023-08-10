from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_data
import pandas as pd

def load_data():
    return fetch_openml(name="boston", version=1, as_frame=True).frame

def display_head(data):
    print(data.head())

def plot_histogram(data, title, figsize=(20, 15)):
    data.hist(bins=30, figsize=figsize)
    plt.suptitle(title)
    plt.show()

def plot_heatmap(data, title, figsize=(10, 10)):
    corr_matrix = data.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()

def plot_scatter_with_regression(data, x_cols, y_col):
    for col in x_cols:
        sns.regplot(x=data[col], y=data[y_col], scatter_kws={'alpha': 0.5})
        plt.xlabel(col)
        plt.ylabel(y_col)
        plt.show()

def compare_histograms(data1, data2, title1, title2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    data1.hist(bins=30, ax=axes[0])
    data2.hist(bins=30, ax=axes[1])
    axes[0].set_title(title1)
    axes[1].set_title(title2)
    plt.show()

if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data()
    boston = load_data()

    # Display processed and raw data
    display_head(X_train_scaled)
    display_head(boston)

    # Compare histograms of raw and processed data
    compare_histograms(boston.drop('MEDV', axis=1), X_train_scaled, 'Raw Data', 'Processed Data')

    # Heatmaps of correlation matrices for raw and processed data
    plot_heatmap(boston, 'Correlation Heatmap of Raw Data')
    plot_heatmap(X_train_scaled, 'Correlation Heatmap of Processed Data')

    # Scatter plots with regression lines for features most correlated with MEDV
    plot_scatter_with_regression(boston, ['RM', 'LSTAT', 'PTRATIO'], 'MEDV')
