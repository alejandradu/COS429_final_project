import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
 
# Load X_resampled and y_resampled data
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/training_pipeline/new_landmark_datasets/landmark_train_resampled_256_heavy_pad_0.7_0.5_0.5.pkl', 'rb') as f:
    X_resampled, y_resampled = pickle.load(f)
    
# load the test
with open("/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/training_pipeline/new_landmark_datasets/landmark_test_heavy_256_pad_0.7_0.5_0.5.pkl", 'rb') as f:
    X_test, y_test = pickle.load(f)

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(100,50,25,100,100,50,25,100,100,50,25,100,100,50,25,100,100,50,25,100,50,25), (100, 100)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'momentum': [0.8, 0.9, 0.99],
    'batch_size': [64, 128, 256]
}

seed = 333
# Initialize the MLPClassifier
nn = MLPClassifier(random_state=seed, max_iter=50000, learning_rate='adaptive')

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=nn, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_resampled, y_resampled)

# print the accuracy of the best model
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the best model: {accuracy}")

# Save the best model
with open('/Users/alejandraduran/Documents/Pton_courses/COS429/COS429_final_project/trained_classifiers/neural_network_4.03am.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

# Print the best parameters and best score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")  # apparently this IS on held out data - will report this score