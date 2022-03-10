import numpy as np
import matplotlib.pyplot as plt
import sys
import time

np.random.seed(1847)

# Get the data
def loadData(file):
    X = []
    with open(file) as f:
        for l in f:
            point = list(map(float, l.split(",")))
            X.append(point)
    return np.array(X)

def loadLabel(file):
    y = []
    with open(file) as f:
        for l in f:
            label = int(l)
            y.append(label)
    return np.array(y)

X_data = loadData(sys.argv[1])
y_data = loadLabel(sys.argv[2])

class SVM:
    def __init__(self, C, eta):
        self.C = C
        self.eta = eta
        self.k = 10
        
    def hingeLoss(self, X, y, W, b):
        loss = 0.5*np.dot(W, W.T)
        for i in range(X.shape[0]):
            vr = y[i]*(np.dot(W, X[i].T) + b)
            loss += self.C*(max(0, (1-vr)))
        return loss[0][0]
    
    def kFold_cross_validation(self, X, y):
        num_samples = X.shape[0]
        step = num_samples // self.k # Size of each fold
        folds = []
        start = 0
        i = 1
        while i <= self.k:
            folds.append((X[start*step:i*step], y[start*step:i*step]))
            start = i
            i += 1
        cur = 0
        accs = []
        # Iterating through each fold
        while cur < self.k:
            testX, testY = folds[cur] # Test set

            # Use the rest (k-1) folds as Training data
            trainX = [] 
            trainY = []
            for idx, fold in enumerate(folds):
                if idx != cur:
                    for x in fold[0]:
                        trainX.append(x)
                    for y in fold[1]:
                        trainY.append(y)
            trainX = np.array(trainX)
            trainY = np.array(trainY)
            nW, nb = self.fit(trainX, trainY)
            y_pred = np.dot(nW, testX.T) + nb # predictions for test set
            y_pred[y_pred >= 0] = 1
            y_pred[y_pred < 0] = -1
            acc = np.mean(testY == y_pred[0, :])
            accs.append(acc)
            cur += 1
        print(np.mean(np.array(accs)))
        print(self.C)
        print(self.eta)
            
    def fit(self, X, y, num_iter=800):
        num_features = X.shape[1]
        num_samples = X.shape[0]
        nW = np.ones((1, num_features))*0.5 # Initialize 
        nb = 0.5
        
        losses =[]
        for itr in range(num_iter):
            l = self.hingeLoss(X, y, nW, nb)
            losses.append(l)
            delta_W, delta_b = 0, 0
            for i in range(num_samples):
                vr = y[i]*(np.dot(nW, X[i].T) + nb)
                if vr < 1:
                    delta_W += self.C*y[i]*X[i] # shape = (1xd)
                    delta_b += self.C*y[i]
            nW = nW - self.eta*nW + self.eta*delta_W
            nb = nb + self.eta*delta_b
        return nW, nb

start = time.time()
svm = SVM(C=0.1, eta=0.001)
svm.kFold_cross_validation(X_data, y_data)
end = time.time()
# print("Time = ", end - start)