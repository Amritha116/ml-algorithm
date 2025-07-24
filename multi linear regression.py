import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
california = fetch_california_housing()

# Select only the features for the 3D plot
X = pd.DataFrame(california.data, columns = california.feature_names)
y = pd.Series(california.target)
# Select only the features for the 3D plot
X_plot_features = X[['MedInc', 'AveRooms']]


X_all = pd.DataFrame(california.data, columns = california.feature_names)

X_all['HousePrice'] = california.target
print("First 5 rows of the dataset :")
print(X_all.head())

# Split the data into training and testing sets using only the selected features
X_train, X_test, y_train, y_test = train_test_split(X_plot_features, y, test_size=0.2, random_state=42)

#Train the model using only the selected features
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions - This is for evaluating the model on the test set,
# using the same features it was trained on.
y_pred = model.predict(X_test)

#Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

#Scatter plot of actual data - Use the selected test features
ax.scatter(X_test['MedInc'], X_test['AveRooms'], y_test, color='blue', label='Actual Data')

#Create a mesh grid for prediction surface
X1_range = np.linspace(X_test['MedInc'].min(), X_test['MedInc'].max(), 100)
X2_range = np.linspace(X_test['AveRooms'].min(), X_test['AveRooms'].max(), 100)
X1,X2 = np.meshgrid(X1_range, X2_range)

#Predict over the mesh grid - The grid now has the correct feature names
grid = pd.DataFrame(np.c_[X1.ravel(), X2.ravel()],columns = ['MedInc', 'AveRooms'])

# Predict using the model trained on only 'MedInc' and 'AveRooms'
Z = model.predict(grid).reshape(X1.shape)

#Plot Regression surface
ax.plot_surface(X1, X2, Z, color='red', alpha=0.5, rstride = 100, cstride = 100, label='Regression Surface')

#Set labels and title
ax.set_xlabel('Median Income')
ax.set_ylabel('Average Rooms')
ax.set_zlabel('House Price')
ax.set_title('Multiple Linear Regression on California Housing Dataset Best Fit Line (3D)')
ax.legend()
plt.show()
