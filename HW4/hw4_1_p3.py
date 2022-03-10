import numpy as np
import sys

def load_data(file):
    data = []
    label = []
    with open(file) as f:
        for line in f:
            if not line:
                continue
            row = line.strip().split(",")
            x = np.array(list((map(float, row[:-1]))))
            y = int(float(row[-1]))
            data.append(x)
            label.append(y)
    label = np.array(label)
    data = np.array(data)
    return data, label

X_train, y_train = load_data(sys.argv[1])
X_test, y_test = load_data(sys.argv[2])

# Change to one-hot coding
def one_hot_convert(data):
    shape = (data.size, 10)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    return one_hot

y_train = one_hot_convert(y_train)
y_test = one_hot_convert(y_test)


class FCLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_diff, lr):
        input_diff = np.dot(output_diff, self.weights.T)
        weights_diff = np.dot(self.input.T, output_diff)

        self.weights -= lr * weights_diff
        self.bias -= lr * output_diff
        return input_diff

class ActivationLayer:
    def __init__(self, activation, activation_dev):
        self.activation = activation
        self.activation_dev = activation_dev
    
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_diff, lr):
        return output_diff * self.activation_dev(self.input)
        

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_dev(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_dev(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

def predict(myFC, input):
    output = input
    for layer in myFC:
        output = layer.forward(output)
    return output

myFC = [
    FCLayer(28 * 28, 128),
    ActivationLayer(sigmoid, sigmoid_dev),
    FCLayer(128, 10),
    ActivationLayer(sigmoid, sigmoid_dev)
]

num_its = 20
lr = 0.1

for it in range(num_its):
    for x, y_true in zip(X_train, y_train):
        x = np.reshape(x, (1, 784))
        output = x
        # print("x shape", x.shape)
        for layer in myFC:
            output = layer.forward(output)

        output_diff = mse_dev(y_true, output)
        for layer in reversed(myFC):
            output_diff = layer.backward(output_diff, lr)

train_acc = sum([np.argmax(y) == np.argmax(predict(myFC, x)) for x, y in zip(X_train, y_train)]) / len(X_train)
test_acc = sum([np.argmax(y) == np.argmax(predict(myFC, x)) for x, y in zip(X_test, y_test)]) / len(X_test)

print(train_acc)
print(test_acc)
print(num_its)
print(lr)
