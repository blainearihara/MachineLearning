# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 17:11:49 2025

@author: blain
"""

#%% Libraries
import numpy as np
import matplotlib.pyplot as plt

#%% Linear Regression Class
class LinearRegression():
    def __init__(self, degree=1, include_bias=True, regularization=None, alpha=0.01):
        """
        Linear Regression with Polynomial Features
        
        Parameters:
        - degree: Polynomial degree for feature expansion
        - include_bias: Whether to include bias term (intercept)
        - regularization: None, 'ridge', or 'lasso' 
        - alpha: Regularization strength
        """
        self.degree = degree
        self.include_bias = include_bias
        self.regularization = regularization
        self.alpha = alpha
        
        # Model parameters
        self.theta = None
        self.feature_names = None
        
    def create_polynomial_features(self, X):
        """
        Create polynomial features up to specified degree
        Similar to your xhat creation but generalized
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Start with original features
        poly_features = X.copy()
        feature_names = [f'x{i}' for i in range(n_features)]
        
        # Add polynomial terms
        for deg in range(2, self.degree + 1):
            for feat_idx in range(n_features):
                new_feature = (X[:, feat_idx] ** deg).reshape(-1, 1)
                poly_features = np.hstack((poly_features, new_feature))
                feature_names.append(f'x{feat_idx}^{deg}')
        
        # Add bias term (constant) at the end if specified
        if self.include_bias:
            bias_column = np.ones((n_samples, 1))
            poly_features = np.hstack((poly_features, bias_column))
            feature_names.append('bias')
            
        return poly_features, feature_names
    
    def fit(self, X, y):
        """
        Fit the linear regression model using normal equation
        """
        # Create polynomial features
        X_poly, self.feature_names = self.create_polynomial_features(X)
        
        # Calculate model parameters using normal equation
        # theta = (X^T * X)^(-1) * X^T * y
        XTX = X_poly.T.dot(X_poly)
        XTy = X_poly.T.dot(y)
        
        # Add regularization if specified
        if self.regularization == 'ridge':
            # Add ridge penalty to diagonal
            XTX += self.alpha * np.eye(XTX.shape[0])
            if self.include_bias:
                # Don't regularize bias term
                XTX[-1, -1] -= self.alpha
        
        try:
            XTX_inv = np.linalg.inv(XTX)
            self.theta = XTX_inv.dot(XTy)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            self.theta = np.linalg.pinv(X_poly).dot(y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if self.theta is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create polynomial features for prediction
        X_poly, _ = self.create_polynomial_features(X)
        
        # Calculate predictions
        return X_poly.dot(self.theta)
    
    def score(self, X, y):
        """
        Calculate R-squared score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_coefficients(self):
        """
        Get model coefficients with feature names
        """
        if self.theta is None:
            return None
        
        coeff_dict = {}
        for i, name in enumerate(self.feature_names):
            coeff_dict[name] = self.theta[i]
        
        return coeff_dict
    
    def plot_fit(self, X, y, title="Linear Regression Fit", num_points=100):
        """
        Plot the fitted model (works for 1D input)
        """
        if X.ndim > 1 and X.shape[1] > 1:
            print("Plotting only available for 1D input data")
            return
        
        if X.ndim > 1:
            X = X.flatten()
        
        # Create smooth curve for plotting
        x_plot = np.linspace(np.min(X), np.max(X), num_points)
        y_plot = self.predict(x_plot.reshape(-1, 1))
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, alpha=0.6, label='Data points')
        plt.plot(x_plot, y_plot, 'r-', linewidth=2, 
                label=f'Polynomial fit (degree={self.degree})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def calculate_mse(self, X, y):
        """
        Calculate Mean Squared Error
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)
    
    def calculate_mae(self, X, y):
        """
        Calculate Mean Absolute Error
        """
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))