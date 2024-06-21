import json

import numpy as np

import data_set
import Neural_Network3

# Generate dataset
num_images = 1
image_size = 10
dataset, labels = data_set.set_imgs, data_set.set_ans

size_input_layer = image_size ** 2  # len(dataset)*len(dataset[0])#image_size ** 2
size_first_hidden_layer = 39  # int(size_input_layer**0.5)#int(size_input_layer / 2)
size_second_hidden_layer = 10  # int(size_first_hidden_layer / 2)
size_output_layer = 3
input_layer = []

for group in dataset:
    new_group = []
    for img in group:
        new_img = []
        for row in img:
            new_img += row
        new_img = np.array(new_img)
        new_group.append(new_img)
    input_layer.append(new_group)

# nn = Neural_Network.NeuralNetwork(.043111,size_input_layer,size_first_hidden_layer,size_second_hidden_layer,size_output_layer)
# nn.train(.00000015432, 2000, input_layer[:num_images], labels)
# nn.train(.95432,2000,input_layer[:num_images],output_layer[:num_images])
# Load the JSON data back into a list
# with open("data.json", "r") as json_file:
#     loaded_data = json.load(json_file)

# Convert the list back to a NumPy array
# w1 = np.array(loaded_data['w1'])
# w2 = np.array(loaded_data['w2'])
# w3 = np.array(loaded_data['w3'])
# b1 = np.array(loaded_data['b1'])
# b2 = np.array(loaded_data['b2'])
# b3 = np.array(loaded_data['b3'])
# nn.w1 = w1
# nn.w2 = w2
# nn.w3 = w3
# nn.b1 = b1
# nn.b2 = b2
# nn.b3 = b3
# nn.train(0.5910019291195432, 10000, input_layer[:num_images], output_layer[:num_images])

# Print the loaded array (should be identical to the original)
# print(loaded_arr)

# pr = nn.predict(input_layer)

nn = Neural_Network3.NeuralNetwork(10**(-3), 100, 48, 21, 3)#5725466701791224
with open("data.json", "r") as json_file:
  loaded_data = json.load(json_file)

# Convert the list back to a NumPy array
w1 = np.array(loaded_data['w1'])
w2 = np.array(loaded_data['w2'])
w3 = np.array(loaded_data['w3'])
nn.w1=w1
nn.w2=w2
nn.w3=w3
# # # b1 = np.array(loaded_data['b1'])
# # # b2 = np.array(loaded_data['b2'])
# # # b3 = np.array(loaded_data['b3'])

# nn.b1=b1
# nn.b2=b2
# nn.b3=b3
# print(0.5149607606646166-0.5152482525204005)
ep = nn.train(245999999908984, 600, input_layer[:], labels)#0.6443239412482759

print()
print(ep)
count_sucses=0
pr = nn.predict(input_layer[:])#47
for i in range(len(pr)):#26%
    for j in range(len(pr[i])):
        if max(pr[i]) == pr[i][j] and i % 3 == j:
            count_sucses += 1
# print('suc= ',(count_sucses/(12*3)))
s = []

for j in range(len(pr)):
   s.append(sum(((pr[j] - labels[j%3]) ** 2)))
data = np.array(s)

mean = np.mean(data)

# if mean < minimum + 0.00001:
#                     print('lr = ', lr, ', a = ', a, 'h1= ', h1, 'h2=', h2)
print(mean)
r = 0
for i in range(len(pr)//3):
    for j in range(3):
        if pr[i+j][j]==max(pr[i+j]):
            print(1,end='')
            r+=1
        else:
            print(0,end='')
        # print(pr[i+j], end='')
    print()
print(len(pr)*3)
print('r in:', r/(len(pr)*3))
print(r)
print(mean)

                # minimum = min(mean, minimum)





# minimum = 5
# for i in range(5):
#     print(i, '/10')
#     for k in range(5):
#         for h1 in range(42,80,11):
#             for h2 in range(5,50,12):
#                 lr=0.2+i * 0.025
#                 a = 0.1+0.1 * k
#
#
#                 nn = Neural_Network3.NeuralNetwork(a, size_input_layer, 82, 3,
#                                                    size_output_layer)
#
#                 nn.train(lr, 60, input_layer[:15], labels)
#                 pr = nn.predict(input_layer[15:])#2721917060078633
#
#                 s = []
#                 # print('pr')
#                 # print(len(pr))
#                 for j in range(len(pr)):
#                     # print('prj')
#                     # print(pr[j])
#                     # print(labels[j%3])
#                     # print(((pr[j] - labels[j%3]) ** 2))
#                     # print(pr[j] - labels[j%3])
#                     s.append(sum(((pr[j] - labels[j%3]) ** 2)))
#                 data = np.array(s)
#                 # print('data')
#                 # print(pr[0])
#                 # print(data)
#                 mean = np.mean(data)
#
#                 if mean < minimum + 0.00001:
#                     print('lr = ', lr, ', a = ', a, 'h1= ', h1, 'h2=', h2)
#                     print(mean)
#                     print(pr)
#                 minimum = min(mean, minimum)
    # print(pr)
# print('minimum=', minimum)
# todo lr 0.32-0.36

# an = output_layer
# an = output_layer[num_images - 5:]


# print(an)

# synapses_input_to_first_hidden_layer = [random.random() for _ in range(size_input_layer * size_first_hidden_layer)]
# synapses_first_hidden_to_second_hidden_layer = [random.random() for _ in
#                                                 range(size_second_hidden_layer * size_first_hidden_layer)]
# synapses_second_hidden_to_output_layer = [random.random() for _ in range(size_output_layer * size_second_hidden_layer)]
#
# input_layer = []
# first_hidden_layer = []
# second_hidden_layer = []
# output_layer = []
# def training_model():
#     for set in dataset:
#         input_layer+=set


# # Save the trained model
# # model.save('a_trained_model.h5')
# model.save('b_trained_model.h5')
