import numpy as np
import sys
with open(sys.argv[1], 'r') as train:
    train = train.readlines()
with open(sys.argv[2], 'r') as test:
    test = test.readlines()
train_features = list()
train_labels = list()
for i in range(len(train)):
    if i == 0:
        continue
    line = train[i]
    line = [float(v) for v in line.split(",")]
    feature, label = line[:-1], int(line[-1])
    train_features.append(feature)
    train_labels.append(label)
true_labels = list()
for e in train_labels:
    ar = np.zeros(10,)
    ar[e] = 1
    true_labels.append(ar)
train_labels = np.asarray(np.matrix(true_labels))
train_features = np.asarray(np.matrix(train_features))
test_features = list()
test_labels = list()
for i in range(len(test)):
    if i == 0:
        continue
    line = test[i]
    line = [float(v) for v in line.split(",")]
    feature, label = line[:-1], int(line[-1])
    test_features.append(feature)
    test_labels.append(label)
true_labels = list()
for e in test_labels:
    ar = np.zeros(10,)
    ar[e] = 1
    true_labels.append(ar)
test_labels = np.asarray(np.matrix(true_labels))
test_features = np.asarray(np.matrix(test_features))
class Fully_Connected_Layer():
    def __init__(self,learning_rate=0.1):

        self.X0 = None
        self.X1 = None
        self.X2 = None
        self.X3 = None
        self.Z1 = None
        self.Z2 = None
        self.Z3 = None
        self.W1 = np.random.randn(128, 784)*0.1
        self.W2 = np.random.randn(64, 128)*0.1
        self.W3 = np.random.randn(10, 64)*0.1
        self.lr = learning_rate
        # we save all parameters in the neural network in this dictionary
    def sigmoid(self, x, derivative=False):
        return 1/(1 + np.exp(-x))   
    def sigmoid_dev(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    def forward(self, Input):
        self.X0 = Input
        self.X1 = np.dot(self.W1, self.X0)
        self.Z1 = self.sigmoid(self.X1)
        self.X2 = np.dot(self.W2, self.X1)
        self.Z2 = self.sigmoid(self.X2)
        self.X3 = np.dot(self.W3, self.Z2)
        self.Z3 = self.sigmoid(self.X3)

        return self.Z3
    def backward(self, Label, output):
        error = 2 * (output - Label) / output.shape[0] * self.sigmoid_dev(self.X3)
        dw3 = np.outer(error, self.Z2)
        error = np.dot(self.W3.T, error) * self.sigmoid_dev(self.X2)
        dw2 = np.outer(error, self.Z1)
        error = np.dot(self.W2.T, error) * self.sigmoid_dev(self.X1)
        dw1 = np.outer(error, self.X0)
        self.W1 -= self.lr * dw1
        self.W2 -= self.lr * dw2
        self.W3 -= self.lr * dw3
    def train(self, Input, Label):
        output = self.forward(Input)
        self.backward(Label, output)
dnn = Fully_Connected_Layer()
for iter in range(20):
    for x,y in zip(train_features, train_labels):
        output = dnn.forward(x)
        dnn.backward(y, output)
predictions = []
predictions_train = []
for x, y in zip(test_features, test_labels):
    output = dnn.forward(x)
    pred = np.argmax(output)
    predictions.append(pred == np.argmax(y))

for x, y in zip(train_features, train_labels):
    output = dnn.forward(x)
    pred = np.argmax(output)
    predictions_train.append(pred == np.argmax(y))

print(np.mean(predictions_train))

print(np.mean(predictions))

print(20)

print(0.1)