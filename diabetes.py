import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('diabetes.csv')

# Feature Engineering
df['pregnant_insulin_product'] = df['Pregnancies'] * df['Insulin']

# Split dataset into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Scale the data
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Define base classifiers
rf_classifier = RandomForestClassifier(random_state=0)
gb_classifier = GradientBoostingClassifier(random_state=0)
knn_classifier = KNeighborsClassifier()

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
    cv=3,  # Reduced cross-validation folds
    stack_method='predict_proba'  # Use predict_proba for meta-classifier input
)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'rf__n_estimators': [50, 100],
    'gb__n_estimators': [50, 100],
    'knn__n_neighbors': [3, 5],
    'final_estimator__C': [0.1, 1.0]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(stacking_classifier, param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train)

# Get the best classifier
best_classifier = grid_search.best_estimator_

# Make predictions
y_pred = best_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100)
