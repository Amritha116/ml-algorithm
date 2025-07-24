import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

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

# Decision Tree with GridSearch
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
param_grid = {
    'max_depth': [3, 4],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

# Output results
print("Best Parameters:", grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot the decision tree
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(best_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Tuned Decision Tree Classifier (ID3)")
plt.show()
