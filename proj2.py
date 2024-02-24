import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Load training data from trainProject.txt
train_data = np.loadtxt('trainProject2.txt', delimiter=',')
X_train_data = train_data[:, 0]
Y_train_data = train_data[:, 1]

# Load test data from testProject2.txt
test_data = np.loadtxt('testProject2.txt', delimiter=',')
X_test_data = test_data[:, 0]
Y_test_data = test_data[:, 1]

# Function to generate features for a given function depth (d)
def generate_features(X, k, d):
    features = [np.ones_like(X)]  # Start with a constant term
    for i in range(1, d + 1):
        features.append(np.sin(i * k * X))
    return np.column_stack(features)

# Function to fit linear regression model using scikit-learn
def fit_linear_regression(X, Y, k, d):
    model = LinearRegression()
    features = generate_features(X, k, d)
    model.fit(features, Y)
    return model

# Function to evaluate and return the mean squared error
def evaluate_model(model, X, Y, k, d):
    features = generate_features(X, k, d)
    Y_pred = model.predict(features)
    mse = mean_squared_error(Y, Y_pred)
    return mse

# Function to implement linear regression from scratch
def custom_linear_regression(X, Y, k, d):
    features = generate_features(X, k, d)
    theta = np.linalg.lstsq(features, Y, rcond=None)[0]
    return theta

# Define function depths
function_depths = [0, 1, 2, 3]
k_value = 0.5

# Part c: Plot the results
plt.figure(figsize=(12, 8))
plt.scatter(X_train_data, Y_train_data, label='Training Data', c='b', s=10)
plt.scatter(X_test_data, Y_test_data, label='Test Data', c='g', s=10)

for d_value in function_depths:
    model = fit_linear_regression(X_train_data, Y_train_data, k_value, d_value)
    X_pred = np.linspace(-3, 3, 1000)  # Adjust the range as needed
    features_pred = generate_features(X_pred, k_value, d_value)
    Y_pred = model.predict(features_pred)
    plt.plot(X_pred, Y_pred, label=f'Depth {d_value}')

plt.legend()
plt.title('Linear Regression Results for Different Function Depths (k=0.5)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Part d: Evaluate regression functions and plot errors
mse_scores_sklearn = []
mse_scores_custom = []

for d_value in function_depths:
    model_sklearn = fit_linear_regression(X_train_data, Y_train_data, k_value, d_value)
    theta_custom = custom_linear_regression(X_train_data, Y_train_data, k_value, d_value)
    
    features_test = generate_features(X_test_data, k_value, d_value)
    Y_pred_sklearn = model_sklearn.predict(features_test)
    Y_pred_custom = np.dot(features_test, theta_custom)
    
    mse_sklearn = mean_squared_error(Y_test_data, Y_pred_sklearn)
    mse_custom = mean_squared_error(Y_test_data, Y_pred_custom)
    
    mse_scores_sklearn.append(mse_sklearn)
    mse_scores_custom.append(mse_custom)

# Plot the errors for different function depths
plt.figure(figsize=(10, 6))
plt.plot(function_depths, mse_scores_sklearn, marker='o', label='Sklearn')
plt.plot(function_depths, mse_scores_custom, marker='x', label='Custom')
plt.xlabel('Function Depth')
plt.ylabel('Mean Squared Error (Test Data)')
plt.title('MSE on Test Data for Different Function Depths')
plt.legend()
plt.show()