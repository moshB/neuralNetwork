import random

import data_set_generator
import Neural_Network

# Generate dataset
num_images = 1
image_size = 10
dataset, labels = data_set_generator.generate_dataset(num_images, image_size)
# print(dataset)
# print(labels)
size_input_layer = image_size ** 2  # len(dataset)*len(dataset[0])#image_size ** 2
size_first_hidden_layer = 70  # int(size_input_layer**0.5)#int(size_input_layer / 2)
size_second_hidden_layer = 35  # int(size_first_hidden_layer / 2)
size_output_layer = 3
input_layer = []
# print(dataset)
# print(labels)
for set in dataset:
    s = []
    for i in set:
        for j in i:
            s.append(j)
    input_layer.append(s)

output_layer = []
for l in labels:
    output_layer.append(l)

nn = Neural_Network.NeuralNetwork(.13111, size_input_layer, size_first_hidden_layer, size_second_hidden_layer,
                                  size_output_layer)
# nn = Neural_Network.NeuralNetwork(.43111,size_input_layer,size_first_hidden_layer,size_second_hidden_layer,size_output_layer)
nn.train(.15432, 2000, input_layer[:num_images], output_layer[:num_images])
# nn.train(.95432,2000,input_layer[:num_images],output_layer[:num_images])

pr = nn.predict(input_layer)
# pr = nn.predict(input_layer[num_images-5:])
an = output_layer
# an = output_layer[num_images-5:]
print(pr)
print(len(pr))
print(an)

synapses_input_to_first_hidden_layer = [random.random() for _ in range(size_input_layer * size_first_hidden_layer)]
synapses_first_hidden_to_second_hidden_layer = [random.random() for _ in
                                                range(size_second_hidden_layer * size_first_hidden_layer)]
synapses_second_hidden_to_output_layer = [random.random() for _ in range(size_output_layer * size_second_hidden_layer)]

input_layer = []
first_hidden_layer = []
second_hidden_layer = []
output_layer = []
# def training_model():
#     for set in dataset:
#         input_layer+=set


# # Save the trained model
# # model.save('a_trained_model.h5')
# model.save('b_trained_model.h5')
