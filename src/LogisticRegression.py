import matplotlib.pyplot as plt
import numpy as np
import math

def showCostEvolution(self):
    if self.plot == True:
        plt.plot(self.cost)
        plt.ylabel('Cost Value')
        plt.show()

def g(z):
    a = 1/(1 + (math.exp(-z)))
    return a

def h(xvector, wvector):
    return g(np.sum(np.multiply(xvector, wvector)))

def cost_fct(wvector, xmatrix, y):
    cost = y * np.log(h(xmatrix, wvector))
    cost += ((1-y) * (np.log(1 - h(xmatrix, wvector))))
    return -1 * np.mean(cost)

class  logreg:
    def __init__(self, args):
        self.plot = args.plot
        self.verbose = args.verbose
        self.cost = []

    def training(self, df, y, epochs, lr):
        wvector = np.matrix((len(df.transpose()) + 1) * [0]) 
        xmatrix = np.hstack((np.matrix(np.ones(df.shape[0])).T, df))

        print(y)
        for i in range(epochs):
            yhat = h(xmatrix, wvector)
            cost = cost_fct(wvector, xmatrix, y)
            test = np.mean(np.matmul(xmatrix, wvector.T))
            print(test)
            #logistic_h = np.mean(1/(1 + np.exp(-1 * test)))
            #first = logistic_h - y.reshape(xmatrix.shape[0], -1)
            #vector = wvector - (lr * np.dot(first, xmatrix))
            self.cost.append(cost)
        showCostEvolution(self)