import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load Parkinsons disease dataset
parkinsons_df = pd.read_csv('ParkinsonsDisease.csv')

# Drop name column
parkinsons_df = parkinsons_df.drop(columns='name', axis=1)

# Separate the data and label into X and y respectively
X = parkinsons_df.drop(columns='status', axis=1)
y = parkinsons_df['status']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Define base classifiers
base_classifiers = [
    ('rf', RandomForestClassifier(random_state=0)),
    ('lr', LogisticRegression(random_state=0, solver='saga', max_iter=1000)),
    ('knn', KNeighborsClassifier())
]

# Define meta-classifier
meta_classifier = LogisticRegression()

# Define Stacking Classifier
stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# Fit the preprocessing pipeline
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Train the stacking classifier
stacking_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred = stacking_classifier.predict(X_test_scaled)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print evaluation metrics
print("Classification Report:")
print(classification_rep)
print("Accuracy:", accuracy)

# Save the trained model
with open('parkinsons_model_stacked.pkl', 'wb') as file:
    pickle.dump(stacking_classifier, file)