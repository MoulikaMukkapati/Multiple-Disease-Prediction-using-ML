import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load dataset
heart_df = pd.read_csv('heart.csv')

# Feature Engineering (if any)
# Example: Create new feature as the product of 'age' and 'chol'
heart_df['age_chol_product'] = heart_df['age'] * heart_df['chol']

# Split dataset into features and target
X = heart_df.drop('target', axis=1)
y = heart_df['target']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# Scale the data
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Define base classifiers
rf_classifier = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=5)
gb_classifier = GradientBoostingClassifier(random_state=0, n_estimators=100, max_depth=3)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Define meta-classifier
meta_classifier = LogisticRegression(random_state=0, solver='saga', max_iter=1000)

# Define Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=[
        ('rf', rf_classifier),
        ('gb', gb_classifier),
        ('knn', knn_classifier)
    ],
    final_estimator=meta_classifier,
    cv=5,  # Cross-validation folds
    stack_method='predict_proba'  # Use predict_proba for meta-classifier input
)

# Train the Stacking Classifier
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
