import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # fixed import
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Iris Dataset
iris = load_iris()
X = iris.data
Y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['Target'] = [iris.target_names[i] for i in Y]
print("First 5 rows of the dataset:")
print(df.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)  # fixed typo: "KNeighboursClassifier" ➜ "KNeighborsClassifier"
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn.predict(X_test_scaled)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_mat)
print("\nClassification Report:\n", classification_rep)

# Visualization (using first two features)
X_vis = iris.data[:, :2]
Y_vis = iris.target
knn_vis = KNeighborsClassifier(n_neighbors=5)
knn_vis.fit(X_vis, Y_vis)

# Create meshgrid
X_min, X_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
Y_min, Y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.1),
                     np.arange(Y_min, Y_max, 0.1))

z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

# Plot decision boundaries and data
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=Y_vis, edgecolors='k', marker='o', s=80, linewidth=1, cmap=plt.cm.Paired)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Nearest Neighbors Classifier (k=5)')
plt.grid(True)
plt.show()
