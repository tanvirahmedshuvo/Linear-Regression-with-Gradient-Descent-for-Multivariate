import numpy as np

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Perform gradient descent for multivariate linear regression.
    
    Args:
        X: numpy array of shape (m, n) where m is the number of samples, and n is the number of features.
        y: numpy array of shape (m, 1) containing the target values.
        learning_rate: Step size for gradient descent.
        iterations: Number of iterations to run gradient descent.
        
    Returns:
        weights: Coefficients for the features (including intercept).
        cost_history: List of cost function values for each iteration.
    """
    m, n = X.shape  # m = number of samples, n = number of features
    X = np.hstack((np.ones((m, 1)), X))  # Add a column of ones for the bias term
    weights = np.zeros((n + 1, 1))  # Initialize weights (including intercept) to zero
    cost_history = []  # To store the cost function values

    for _ in range(iterations):
        # Predictions
        y_pred = np.dot(X, weights)
        # Calculate error
        error = y_pred - y
        # Compute the gradient
        gradient = (1 / m) * np.dot(X.T, error)
        # Update the weights
        weights -= learning_rate * gradient
        # Compute and store the cost
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)
    
    return weights, cost_history

# Example usage
if __name__ == "__main__":
    # Sample dataset (House size, Number of rooms, Price)
    # Features: [size (sq ft), number of rooms]
    X = np.array([[2104, 3],
                  [1600, 3],
                  [2400, 3],
                  [1416, 2],
                  [3000, 4]])

    # Target: House prices
    y = np.array([[400],
                  [330],
                  [369],
                  [232],
                  [540]])

    # Perform gradient descent
    weights, cost_history = gradient_descent(X, y, learning_rate=0.0000001, iterations=1000)

    # Print results
    print("Final weights (including intercept):")
    print(weights)

    print("\nFinal cost:", cost_history[-1])