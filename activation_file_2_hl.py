from matplotlib import pyplot as plt

from sets import prepare_data
from sets import prepare_random_data
from Neural_Network_2_hidden_layer import NeuralNetwork

input_layer, labels = prepare_data()
print(f'Input shape: {input_layer.shape}')
print(f'Labels shape: {labels.shape}')
split = 5 * 3
nn = NeuralNetwork(learning_rate=0.1, input_size=100, hidden_size1=90, hidden_size2=32, output_size=1)
nn.train(input_layer[:split], labels[:split], epochs=1)
predictions = nn.predict(input_layer[split:])  # [1,0,0]<-[0.1,0.5,0.8]

count_success = 0
print(labels)
for i in range(len(predictions)):
    mod = i % 3
    if labels[i][0] == 1 / 6 and predictions[i][0] < 1 / 3:
        count_success += 1
    elif labels[i][0] == 0.5 and 2 / 3 > predictions[i][0] > 1 / 3:
        count_success += 1
    elif labels[i][0] == 5 / 6 and predictions[i][0] > 2 / 3:
        count_success += 1

accuracy = count_success / len(predictions)
print(f'Accuracy: {accuracy:.2f}')

counter = 1
x=[]
y=[]
x.append(counter)
y.append(accuracy * 100)
while accuracy < 1:
    counter += 1
    nn.train(input_layer[:split], labels[:split], epochs=1)
    predictions = nn.predict(input_layer[:split])
    print('circle')
    count_success = 0
    for i in range(len(predictions)):
        mod = i % 3
        if labels[i][0] == 1 / 6 and predictions[i][0] < 1 / 3:
            count_success += 1
        elif labels[i][0] == 0.5 and 2 / 3 > predictions[i][0] > 1 / 3:
            count_success += 1
        elif labels[i][0] == 5 / 6 and predictions[i][0] > 2 / 3:
            count_success += 1

    accuracy = count_success / len(predictions)
    x.append(counter)
    y.append(accuracy*100)
    print(f'Accuracy: {accuracy:.2f}')

print(counter)

# לבדוק אחוזים על יתר הקבוצות

predictions = nn.predict(input_layer[split:])  # [1,0,0][0,0.39,0]

count_success = 0
for i in range(len(predictions)):
    mod = i % 3
    if labels[i + split ][0] == 1 / 6 and predictions[i][0] < 1 / 3:
        count_success += 1
    elif labels[i + split][0] == 0.5 and 2 / 3 > predictions[i][0] > 1 / 3:
        count_success += 1
    elif labels[i + split ][0] == 5 / 6 and predictions[i][0] > 2 / 3:
        count_success += 1

accuracy = count_success / len(predictions)
print(f'Accuracy: {accuracy:.2f}')
plt.plot(x, y)
plt.xlabel("מספר חזרות")
plt.ylabel("אחוז הצלחה")
plt.grid(True)
plt.show()
