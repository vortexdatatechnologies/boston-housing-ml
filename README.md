# Boston Housing ML
The Boston Housing database is a famous dataset in the field of machine learning and data science. It was originally published by Harrison and Rubinfeld in 1978. This database contains information collected by the U.S. Census Service about housing in the Boston, Massachusetts area.

This database includes 506 instances (i.e., rows of data) and 14 attributes (or characteristics). The attributes are:

- CRIM: Crime rate per capita by city.

- ZN: Proportion of residential land zoned for lots larger than 25,000 square feet.

- INDUS: Proportion of non-retail commercial acreage by city.

- CHAS: Charles River dummy variable (= 1 if the tract borders the river; 0 otherwise).

- NOX: Concentration of nitric oxides (parts per 10 million).

- RM: Average number of rooms per dwelling.

- AGE: Proportion of owner-occupied units built before 1940.

- DIS: Weighted distances to five Boston employment centers.

- RAD: Radial highway accessibility index.

- TAX: Full-value property tax rate per $10,000.

- PTRATIO: Pupil-teacher ratio by city.

- B: 1000(Bk - 0.63)^2 where Bk is the proportion of people of color by city.

- LSTAT: Least state % of population.

- MEDV: Median value of owner-occupied homes in $1000s.

The goal with this database is usually to predict the median value of houses (MEDV) as a function of the other characteristics. Therefore, it is a commonly used dataset for regression problems.

## Visualization
This code parses the data from the dataset. It obtains the dataset from the OpenML repository and visualizes the data using various techniques. Next, a histogram is created for each column of the dataset and a correlation matrix is calculated to visualize the correlation between the different features.
Finally, scatter plots are created for the features most correlated with the target variable.
Overall, this code provides a basic analysis of the Boston Housing dataset by visualizing the distribution of the data, exploring the correlation between features, and examining the relationship between specific features and the target variable.
