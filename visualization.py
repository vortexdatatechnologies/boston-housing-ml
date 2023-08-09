from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import preprocess_data

train_set_scaled, test_set_scaled, train_set_labels, test_set_labels = preprocess_data()
# Importación y preprocesamiento como lo tienes
# ...

# Mostrar la versión procesada de los datos
print(train_set_scaled.head())

# Crear un histograma para cada columna en los datos procesados
train_set_scaled.hist(bins=30, figsize=(20, 15))
plt.suptitle('Histograms of Preprocessed Data')
plt.show()

# Crear una matriz de correlación para los datos procesados
corr_matrix_processed = train_set_scaled.corr()

# Dibujar un mapa de calor de la matriz de correlación para los datos procesados
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix_processed, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Preprocessed Data')
plt.show()

# ... (Resto del código para visualizar datos originales)

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
for col in ['RM', 'LSTAT', 'PTRATIO']:
    plt.scatter(boston[col], boston['MEDV'])
    plt.xlabel(col)
    plt.ylabel('MEDV')
    plt.show()
