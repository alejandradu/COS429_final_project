import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Load X_resampled and y_resampled data
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/training_pipeline/landmark_datasets/landmark_train_resampled_256_DROP_0.7_0.5_0.5.pkl', 'rb') as f:
    X_resampled, y_resampled = pickle.load(f)

# Define the parameter grid
param_grid = {
    'alpha': [0.001, 0.1, 1],
    'learning_rate_init': [0.001, 0.01],
    'momentum': [0.9,1.5,2.0]
}

seed = 333
# Initialize the MLPClassifier
nn = MLPClassifier(random_state=seed, max_iter=5000, learning_rate='adaptive', hidden_layer_sizes=(100,100,47,100,100))

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=nn, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_resampled, y_resampled)

# print the accuracy of the best model
y_pred = grid_search.predict(X_resampled)
accuracy = accuracy_score(y_resampled, y_pred)
print(f"Accuracy of the best model: {accuracy}")

# Save the best model
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/trained_classifiers/neural_network_best_drop2.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

# Print the best parameters and best score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")