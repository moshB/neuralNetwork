import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate, input_size, hidden_size, output_size):
        self.lr = learning_rate
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Weight initialization
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        # self.w3 = np.random.randn(hidden_size2, output_size) * 0.01
        # self.b3 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)
    def dif_sigmoid(self, x):
        return np.exp(- x) / ((1 + np.exp(- x)) ** 2)

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        # self.z3 = np.dot(self.a2, self.w3) + self.b3
        # self.a3 = self.sigmoid(self.z3)
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]

        self.dz2 = output - y
        # print(f'dz3 shape: {self.dz3.shape}')
        self.dw2 = (1 / m) * np.dot(self.a1.T, self.dz2)
        # print(f'dw3 shape: {self.dw3.shape}')
        self.db2 = (1 / m) * np.sum(self.dz2, axis=0, keepdims=True)
        # print(f'db3 shape: {self.db3.shape}')

        # self.dz2 = np.dot(self.dz3, self.w3.T) * self.dif_sigmoid(self.z2)
        # # print(f'dz2 shape: {self.dz2.shape}')
        # self.dw2 = (1 / m) * np.dot(self.a1.T, self.dz2)
        # # print(f'dw2 shape: {self.dw2.shape}')
        # self.db2 = (1 / m) * np.sum(self.dz2, axis=0, keepdims=True)
        # # print(f'db2 shape: {self.db2.shape}')

        self.dz1 = np.dot(self.dz2, self.w2.T) * self.dif_sigmoid(self.z1)
        # print(f'dz1 shape: {self.dz1.shape}')
        self.dw1 = (1 / m) * np.dot(X.T, self.dz1)
        # print(f'dw1 shape: {self.dw1.shape}')
        self.db1 = (1 / m) * np.sum(self.dz1, axis=0, keepdims=True)
        # print(f'db1 shape: {self.db1.shape}')

        # Update weights
        self.w1 -= self.lr * self.dw1
        self.b1 -= self.lr * self.db1
        self.w2 -= self.lr * self.dw2
        self.b2 -= self.lr * self.db2
        # self.w3 -= self.lr * self.dw3
        # self.b3 -= self.lr * self.db3

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)
