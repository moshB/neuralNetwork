import random

import data_set_generator






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


# Generate dataset
num_images = 1000
image_size = 100
dataset, labels = data_set_generator.generate_dataset(num_images, image_size)

size_input_layer= image_size ** 2
size_first_hidden_layer=int(size_input_layer / 2)
size_second_hidden_layer=int(size_first_hidden_layer / 2)
size_output_layer=1

synapses_input_to_first_hidden_layer = [random.random() for _ in range(size_input_layer * size_first_hidden_layer)]
synapses_first_hidden_to_second_hidden_layer= [random.random() for _ in range(size_second_hidden_layer * size_first_hidden_layer)]
synapses_second_hidden_to_output_layer= [random.random() for _ in range(size_output_layer * size_second_hidden_layer)]

input_layer= []
first_hidden_layer=[]
second_hidden_layer=[]
output_layer=[]
# def training_model():
#     for set in dataset:
#         input_layer+=set












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