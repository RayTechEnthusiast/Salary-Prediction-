import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def get_user_input():
    years = np.array([1, 2, 3, 4]).reshape(-1, 1)  # Define the years of experience as input features
    salaries = []  # List to store salary values

    for year in range(1, 5):  # Loop through each year
        while True:
            try:
                salary = float(input(f"Enter salary for year {year}: "))  # Take user input for salary
                if salary <= 0:
                    print("Salary must be greater than zero. Please enter a valid amount.")
                    continue  # Ensure salary is a positive value
                salaries.append(salary)
                break  # Exit loop once valid salary is entered
            except ValueError:
                print("Invalid input. Please enter a numeric value.")  # Handle invalid inputs

    return years, np.array(salaries).reshape(-1, 1)  # Return years and salaries as NumPy arrays

def predict_and_plot(years, salaries, degree=3, k=4):  # Adjusted k to avoid ValueError
    log_salaries = np.log(salaries)  # Apply log transformation to salaries

    poly = PolynomialFeatures(degree=degree)  # Create polynomial features of given degree
    years_poly = poly.fit_transform(years)  # Transform input years into polynomial features

    k = min(k, len(years))  # Ensure k is not greater than the number of samples
    kf = KFold(n_splits=k, shuffle=True, random_state=42)  # Initialize k-fold cross-validation
    errors = []  # List to store validation errors

    for train_index, test_index in kf.split(years_poly):  # Loop through k-fold splits
        X_train, X_test = years_poly[train_index], years_poly[test_index]  # Split data into training and testing sets
        y_train, y_test = log_salaries[train_index], log_salaries[test_index]  # Split salaries accordingly

        model = LinearRegression()  # Initialize linear regression model
        model.fit(X_train, y_train)  # Train model on training data

        y_pred = model.predict(X_test)  # Predict log salaries for test set
        error = mean_squared_error(y_test, y_pred)  # Compute mean squared error
        errors.append(error)  # Store error value

    avg_error = np.mean(errors)  # Compute average cross-validation error
    print(f"Average Cross-Validation Error: {avg_error:.5f}")  # Print validation error

    model.fit(years_poly, log_salaries)  # Re-train model on entire dataset

    future_years = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)  # Define years for future predictions
    future_years_poly = poly.transform(future_years)  # Transform future years using polynomial features
    predicted_log_salaries = model.predict(future_years_poly)  # Predict log salaries for future years
    predicted_salaries = np.exp(predicted_log_salaries)  # Convert predicted log salaries back to normal scale

    plt.scatter(years, salaries, color='black', marker='o', label='Actual Salaries')  # Plot actual salaries
    plt.plot(future_years, predicted_salaries, color='blue', linestyle='dashed', label='Predicted Salaries')  # Plot predictions

    plt.xlabel('Years')  # Label x-axis
    plt.ylabel('Salary')  # Label y-axis
    plt.title(f'Salary Prediction with Polynomial Regression (Degree {degree})')  # Title for the plot
    plt.yscale('log')  # Use logarithmic scale for salary values
    plt.legend()  # Display legend
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add grid lines
    plt.show()  # Display the plot

if __name__ == "__main__":
    years, salaries = get_user_input()  # Get user input for years and salaries
    predict_and_plot(years, salaries, degree=3, k=4)  # Perform prediction and plot results with k-fold validation
