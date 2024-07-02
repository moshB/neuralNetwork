import numpy as np
from sets import prepare_data1
from sets import prepare_random_data
from Neural_Network_2_hidden_layer import NeuralNetwork

input_layer, labels = prepare_random_data()

print(f'Input shape: {input_layer.shape}')
print(f'Labels shape: {labels.shape}')
split=15*3
nn = NeuralNetwork(learning_rate=2.999, input_size=100, hidden_size1=90, hidden_size2=32, output_size=1)
nn.train(input_layer[:split], labels[:split], epochs=1)
# print(len(input_layer))
predictions = nn.predict(input_layer[split:])#[1,0,0]<-[0.1,0.5,0.8]

count_success = 0
print(labels)
for i in range(len(predictions)):
    mod = i % 3
    if labels[i][0] == 1/6 and predictions[i][0] < 1 / 3:
        count_success += 1
    elif labels[i][0] == 0.5 and predictions[i][0] < 2 / 3 and predictions[i][0] > 1 / 3:
        count_success += 1
    elif labels[i][0] == 5/6 and predictions[i][0] > 2 / 3:
        count_success += 1

accuracy = count_success / len(predictions)
print(f'Accuracy: {accuracy:.2f}')


counter = 1
while accuracy < 1:
    counter += 1
    # nn.lr = nn.lr*0.98
    nn.train(input_layer[:split], labels[:split], epochs=1)
    # print(len(input_layer))
    predictions = nn.predict(input_layer[:split])
    print('circle')
    count_success = 0
    for i in range(len(predictions)):
        mod = i % 3
        if labels[i][0] == 1/6 and predictions[i][0] < 1 / 3:
            count_success += 1
        elif labels[i][0] == 0.5 and predictions[i][0] < 2 / 3 and predictions[i][0] > 1 / 3:
            count_success += 1
        elif labels[i][0] == 5/6 and predictions[i][0] > 2 / 3:
            count_success += 1

    accuracy = count_success / len(predictions)
    print(f'Accuracy: {accuracy:.2f}')

print(counter)

#todo לבדוק אחוזים על יתר הקבוצות

predictions = nn.predict(input_layer[split:])  # [1,0,0][0,0.39,0]
# print(predictions)

count_success = 0
for i in range(len(predictions)):
    mod = i % 3
    if labels[i+split-1][0] == 1/6 and predictions[i][0] < 1 / 3:
        count_success += 1
    elif labels[i+split-1][0] == 0.5 and predictions[i][0] < 2 / 3 and predictions[i][0] > 1 / 3:
        count_success += 1
    elif labels[i+split-1][0] == 5/6 and predictions[i][0] > 2 / 3:
        count_success += 1

accuracy = count_success / len(predictions)
print(f'Accuracy: {accuracy:.2f}')
