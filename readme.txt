Polynomial Regression with Scikit-Learn and Custom Implementation
This project implements polynomial regression using both Scikit-Learn and custom implementation in Python. It explores the effect of polynomial function depth on regression performance and compares the results between the two implementations.

Overview
The polynomial_regression.py script included in this repository loads training and test data from text files, fits polynomial regression models of varying depths using Scikit-Learn, and evaluates the models' performance. Additionally, it implements polynomial regression from scratch to compare results.

Usage
Clone the repository to your local machine:
git clone https://github.com/your_username/polynomial-regression.git

Ensure you have the necessary dependencies installed:
Copy code
pip install numpy matplotlib scikit-learn

Run the script:
python polynomial_regression.py
Check the generated plots to observe the regression results and mean squared errors for different function depths.

Requirements
Python 3.x
NumPy
Matplotlib
Scikit-Learn

Results
The script will produce two plots:
Regression Results for Different Function Depths: This plot shows the regression curves fitted to the training data for varying function depths, along with the test data points.
Mean Squared Error (MSE) on Test Data for Different Function Depths: This plot compares the MSE scores of the Scikit-Learn and custom implementations for different function depths.

