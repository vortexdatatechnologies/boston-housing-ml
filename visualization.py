from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Boston Housing dataset
boston_dataset = fetch_openml(name="boston", version=1, as_frame=True)
boston = boston_dataset.frame
target = boston['MEDV']

# Display the values of the target variable
print(target.head())

# Display the first few rows of the dataset
print(boston.head())

# Create a histogram for each column
boston.hist(bins=30, figsize=(20, 15))
plt.show()

# Create a correlation matrix
corr_matrix = boston.corr()

# Draw a heatmap of the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Create scatter plots of the features most correlated with MEDV
for col in ['rm', 'lstat', 'ptratio']:
    plt.scatter(boston[col], boston['medv'])
    plt.xlabel(col)
    plt.ylabel('medv')
    plt.show()
