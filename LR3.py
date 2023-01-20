# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:09:43 2022

@author: CamilaMusina
"""

from math import exp
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

X1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
X2 = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
X3 = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
X4 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

X_test = [[X1[i], X2[i], X3[i], X4[i]] for i in range(16)]

C = np.array([[0, 1, 1, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]])

EPOCH_NUMBER = 5
BIG_EPOCH_NUMBER = 120
lr = 0.3

def simulated_boolean_function(x1, x2, x3, x4):
    return int((not x1 or not x2 or not x3) and (not x2 or not x3 or x4))

def loss(y_true, y_predicted):
    return y_true - y_predicted

def get_predict(net, X_test):
    y_predicted = []
    y_predicted = [net.forward(x) for x in X_test]
    return np.array(y_predicted)

class RadialBasisFunctionNeuron:
    def __init__(self, center):
        self.center = center
    
    def get_phi(self, X):
        summary = np.sum((X - self.center) ** 2)
        return exp(-summary)

class NeuronNet:
    
    def __init__(self, neurons_count, centers, flag : bool):
        self.neurons_count = neurons_count
        self.neuron = []
        for i in range(self.neurons_count):
            self.neuron.append(RadialBasisFunctionNeuron(centers[i]))
        self.W = np.zeros(shape=neurons_count)
        self.b = 0
        self.flag = flag
    
    def forward(self, x):
        self.phi = []
        for i in range(self.neurons_count):
            self.phi.append(self.neuron[i].get_phi(x))
        self.phi = np.array(self.phi)
        self.net = np.dot(self.W, self.phi) + self.b
        if self.flag:
            self.softsign = 0.5 * (self.net / (1 + np.abs(self.net)) + 1)
            return int(self.softsign > 0.5)
        else:
            return int(self.net >= 0)
    
    def backward(self, delta, lr=0.3):
        self.dz = 1
        if self.flag:
            self.dz = 0.5 / (1 + np.abs(self.softsign) ** 2)
        self.dW = np.dot(lr * delta * self.dz, self.phi)
        self.db = lr * delta * self.dz
        self.W = self.W + self.dW
        self.b = self.b + self.db
        
    def show_plot(self, L_iter : list):
        fig, ax = plt.subplots()
        ax.plot(L_iter)
        ax.set_xlabel('epoch number')
        ax.set_ylabel('error')
        ax.set_title('Error(epoch)')
        plt.grid()
        
    def learning(self):
        L_iter = []
        size_of_train = 16
        for epoch in range(EPOCH_NUMBER):
            error = 0.
            print("\n\tEpoch ", epoch)
            Y_predicted = []
            for i in range(size_of_train):
                x = [X1[i], X2[i], X3[i], X4[i]]
                y_true = Y[i]
                y_predicted = self.forward(x)
                delta = loss(y_true, y_predicted)
                error += np.abs(delta)
                self.backward(delta, lr)
                Y_predicted.append(y_predicted)
            L_iter.append(error)
            print("W = ", net.W, 'b = ', self.b)
            print("Y_pred = ", np.array(Y_predicted))
            print("Y_true = ", Y)
            print("E = ", error)
            if (error == 0):
                break
            
        self.show_plot(L_iter)
        """
        fig, ax = plt.subplots()
        ax.plot(L_iter)
        ax.set_xlabel('epoch number')
        ax.set_ylabel('error')
        ax.set_title('Error(epoch)')
        plt.grid()
        
        """

    def learning_partly(self):
        for number_of_variables in range(15, 0, -1):
            combs = tuple(combinations(range(16), number_of_variables))
            for idxs in combs:
                net = NeuronNet(3, C, self.flag)        
                L_iter = []
                for epoch in range(BIG_EPOCH_NUMBER):
                    error_on_train = 0.
                    Y_predicted = []
                    for i in idxs:
                        x = [X1[i], X2[i], X3[i], X4[i]]
                        y_true = Y[i]
                        y_predicted = net.forward(x)
                        delta = loss(y_true, y_predicted)
                        error_on_train += abs(delta)
                        net.backward(delta, lr)
                        Y_predicted.append(y_predicted)
                    error_on_test = np.sum(loss(get_predict(net, X_test), Y))
                    L_iter.append(error_on_test)
                    if (error_on_test == 0):
                        break
                        
                if L_iter[-1] == 0:
                    print("Combination of size ", number_of_variables, " found. Indexes : ", idxs)
                    break
                
    def min_set_learn(self):
        L_iter = []
        idxs = [3, 14]
        if self.flag:
            idxs = [0, 14]
        for epoch in range(EPOCH_NUMBER):
            error_on_train = 0.
            print("\n\tEpoch ", epoch)
            for i in idxs:
                x = [X1[i], X2[i], X3[i], X4[i]]
                y_true = Y[i]
                y_predicted = self.forward(x)
                delta = loss(y_true, y_predicted)
                error_on_train += abs(delta)
                self.backward(delta, lr)
            Y_predicted = get_predict(self, X_test)
            error_on_test = np.sum(np.abs(loss(Y_predicted, Y)))
            L_iter.append(error_on_test)
            print("W = ", self.W)
            print("Y = ", Y_predicted)
            print("E = ", error_on_test)
            if (error_on_test == 0):
                break

        print("\nfinal W = ", self.W)
        self.show_plot(L_iter)
        """
        fig1, ax1 = plt.subplots()
        ax1.plot(L_iter)
        ax1.set_xlabel('epoch number')
        ax1.set_ylabel('error')
        ax1.set_title('Error(epoch)')
        plt.grid()
        plt.show()
        """

Y = np.array([simulated_boolean_function(X1[i], X2[i], X3[i], X4[i]) for i in range(16)])


if __name__ == '__main__':
    net = NeuronNet(3, C, False)
    net.learning()
    #net.learning_partly()
    net.min_set_learn()

    net1 = NeuronNet(3, C, True)
    net1.learning()
    #net1.learning_partly()
    net1.min_set_learn()

