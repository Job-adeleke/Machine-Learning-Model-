import numpy as np

#logistic regression 

class Logistic_Regression():

    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations  

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)  
        self.b = 0
        self.X = X
        self.Y = Y
        
        # Implementing gradient descent 
        for i in range(self.no_of_iterations):
            self.update_weights()
        
    def update_weights(self):
        # Y_hat formula (sigmoid function)
        Y_hat = 1 / (1 + np.exp(- (self.X.dot(self.w) + self.b)))  # wX + b

        # Derivatives 
        dw = (1/self.m) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1/self.m) * np.sum(Y_hat - self.Y)
        
        # Updating the weight & bias using gradient descent 
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):  # Added X as a parameter to predict method
        Y_pred = 1 / (1 + np.exp(- (X.dot(self.w) + self.b)))
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred