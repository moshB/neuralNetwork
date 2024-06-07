import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Initialize weights and biases with random values
        self.w1 = np.random.rand(input_size, hidden_size1)
        self.b1 = np.zeros((1, hidden_size1))  # Bias for first hidden layer
        self.w2 = np.random.rand(hidden_size1, hidden_size2)
        self.b2 = np.zeros((1, hidden_size2))  # Bias for second hidden layer
        self.w3 = np.random.rand(hidden_size2, output_size)
        self.b3 = np.zeros((1, output_size))  # Bias for output layer

        # Choose an activation function (e.g., sigmoid, ReLU)
        self.activation_func = self.sigmoid  # Placeholder, replace with your desired function

    def sigmoid(self, x):
        """Sigmoid activation function (example)"""
        return 1 / (1 + np.exp(-x))

    def dif_sigmoid(self, x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def predict(self, x):
        """Forward pass to calculate output"""
        z1 = np.dot(x, self.w1) + self.b1  # Weighted sum at hidden layer 1
        a1 = self.activation_func(z1)  # Apply activation function

        z2 = np.dot(a1, self.w2) + self.b2  # Weighted sum at hidden layer 2
        a2 = self.activation_func(z2)  # Apply activation function

        z3 = np.dot(a2, self.w3) + self.b3  # Weighted sum at output layer
        output = self.activation_func(z3)  # Apply activation function (can be different for output)

        return output

    def train(self, learning_rate, epochs, X_train, y_train):
        for i in range(len(X_train)):
            print(i, end=" ")
            x = np.array([X_train[i]])

            # print(x)
            # Forward pass
            z1 = np.dot(x, self.w1) + self.b1
            a1 = self.activation_func(z1)

            z2 = np.dot(a1, self.w2) + self.b2
            a2 = self.activation_func(z2)

            z3 = np.dot(a2, self.w3) + self.b3
            output = self.activation_func(z3)

            d3 = self.dif_sigmoid(z3) * (y_train[i] - output)
            self.w3 += np.dot(a2.T, d3)
            d2 = self.dif_sigmoid(z2) * (sum(sum(d3)))
            self.w2 += np.dot(a1.T, d2)
            d1 = self.dif_sigmoid(z1) * (sum(sum(d2)))
            self.w1 += np.dot(x.T, d1)
        print()
