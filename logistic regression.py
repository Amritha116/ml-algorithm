import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate synthetic dataset
X, y = make_classification(
    n_samples=2000, n_features=2, n_classes=2,
    n_informative=2, n_redundant=0, class_sep=2.0, random_state=42
)

# Data before scaling
df_before = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df_before['Target'] = y
print("First 5 rows of the dataset (before scaling):")
print(df_before.head())

# Initial train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Show scaled training data
df_after = pd.DataFrame(X_train_scaled, columns=['Feature 1_scaled', 'Feature 2_scaled'])
df_after['Target'] = y_train
print("\nFirst 5 rows of the scaled training dataset:")
print(df_after.head())

# Logistic Regression model
model = LogisticRegression(C=10.0, solver='lbfgs', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Prediction & evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_mat)
print("\nClassification Report:\n", classification_rep)

# Plotting decision boundary
if X_train_scaled.shape[1] == 2:
    X_all = np.vstack((X_train_scaled, X_test_scaled))
    y_all = np.hstack((y_train, y_test))
    h = 0.02
    X_min, X_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    Y_min, Y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(X_min, X_max, h),
        np.arange(Y_min, Y_max, h)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, edgecolors='k', s=80, cmap='viridis')
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title("Logistic Regression Decision Boundary (High Accuracy)")
    plt.grid(True)
    plt.show()
else:
    print("Cannot plot decision boundary for data with more than 2 features.")
