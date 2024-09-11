import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Load Iris dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Display the first few rows
data.head()
# Check for missing values
print(data.isnull().sum())

# Normalize features (if necessary)

scaler = StandardScaler()
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# Check normalized data
data.head()

X = data.drop(columns='species')
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train the Decision Tree model

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
# Sample new data for prediction
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Example input
new_data_scaled = scaler.transform(new_data)  # Scale the input
prediction = model.predict(new_data_scaled)
print(f'Predicted Class: {iris.target_names[prediction[0]]}')
# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(conf_matrix).plot()
plt.show()

# Feature Importance (for Decision Tree)

importances = model.feature_importances_
features = iris.feature_names
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.show()
