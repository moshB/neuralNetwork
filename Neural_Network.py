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

        """Train the network using backpropagation"""
        # for epoch in range(epochs):
        #     for i, x in enumerate(X_train):  # Iterate through each training example
        #         # Forward pass
        #         z1 = np.dot(x, self.w1) + self.b1
        #         a1 = self.activation_func(z1)
        #
        #         z2 = np.dot(a1, self.w2) + self.b2
        #         a2 = self.activation_func(z2)
        #
        #         z3 = np.dot(a2, self.w3) + self.b3
        #         output = self.activation_func(z3)
        #
        #         # Calculate error (adjust based on your loss function, e.g., cross-entropy)
        #         error = y_train[i] - output
        #         d3 = self.dif_sigmoid(z3)*(y_train[i]-output)
        #         self.w3 += np.dot(a2.T, d3)
        #         d2 = self.dif_sigmoid(z2)*(sum(sum(d3)))
        #         self.w2+=np.dot(a1.T, d2)
        #         d1 = self.dif_sigmoid(z1)*(sum(sum(d2)))
        #         self.w1+=np.dot(x.T, d1)
        #

        # Backpropagation
        # delta3 = error * self.activation_func(z3)#, derivative=True)  # Output layer delta
        # delta2 = np.dot(delta3, self.w3.T) * self.activation_func(z2)#, derivative=True)  # Hidden layer 2 delta
        # delta1 = np.dot(delta2, self.w2.T) * self.activation_func(z1)#, derivative=True)  # Hidden layer 1 delta
        #
        # # Update weights and biases with gradients
        # self.w3 -= learning_rate * np.dot(a2.T, delta3)
        # print(sum(delta3))
        # # print(learning_rate * delta3)
        # # print(self.b3)
        # self.b3 -= learning_rate * delta3
        #
        # self.w2 -= learning_rate * np.dot(a1.T, delta2)
        # self.b2 -= learning_rate * delta2
        #
        # self.w1 -= learning_rate * np.dot(x.T, delta1)
        # self.b1 -= learning_rate * delta1

# def split_dataset(dataset, labels, test_size=0.2, validation_size=0.2, random_state=None):
#     # First, split the dataset into training and test sets
#     train_images, test_images, train_labels, test_labels = train_test_split(
#         dataset, labels, test_size=test_size, random_state=random_state)
#
#     # Then, split the training set into training and validation sets
#     train_images, val_images, train_labels, val_labels = train_test_split(
#         train_images, train_labels, test_size=validation_size / (1 - test_size), random_state=random_state)
#
#     return train_images, val_images, test_images, train_labels, val_labels, test_labels

#
# # Generate dataset
# num_images = 1000
# image_size = 100
# dataset, labels = data_set_generator.generate_dataset(num_images, image_size)
#
# size_input_layer= image_size ** 2
# size_first_hidden_layer=int(size_input_layer / 2)
# size_second_hidden_layer=int(size_first_hidden_layer / 2)
# size_output_layer=1
#
# synapses_input_to_first_hidden_layer = [random.random() for _ in range(size_input_layer * size_first_hidden_layer)]
# synapses_first_hidden_to_second_hidden_layer= [random.random() for _ in range(size_second_hidden_layer * size_first_hidden_layer)]
# synapses_second_hidden_to_output_layer= [random.random() for _ in range(size_output_layer * size_second_hidden_layer)]
#
# input_layer= []
# first_hidden_layer=[]
# second_hidden_layer=[]
# output_layer=[]
# def training_model():
#     for set in dataset:
# input_layer+=set


#
#
#
#
#
#
#
# # Split the dataset into training, validation, and test sets
# (train_images, val_images, test_images, train_labels, val_labels, test_labels) =\
#     split_dataset(dataset, labels, test_size=0.2, validation_size=0.2, random_state=42)
#
# # Preprocess the datasets
# train_images = train_images.reshape(-1, image_size, image_size, 1).astype('float32') / 255.0
# val_images = val_images.reshape(-1, image_size, image_size, 1).astype('float32') / 255.0
# test_images = test_images.reshape(-1, image_size, image_size, 1).astype('float32') / 255.0
#
# # Define a simple CNN model
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(3, activation='softmax')
# ])
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
#
# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(test_images, test_labels)
# print("Test Loss:", test_loss)
# print("Test Accuracy:", test_accuracy)
#
# # Save the trained model
# # model.save('a_trained_model.h5')
# model.save('b_trained_model.h5')
