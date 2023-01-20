# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 14:01:14 2022

@author: CamilaMusina
"""
import matplotlib.pyplot as plt
import math

class NeuronalNetwork:
    
    def __init__(self, a, b, M, k, p = 4):
        self.a = a
        self.b = b
        self.M = M
        self.learnK = k
        
        self.N = 20
        self.p = p
        
        self.w = [0] * (self.p + 1)
        self.epoch = []
        self.E = []
        self.eps = 0.0
        
        self.realX = []
        self.realY = []
        self.predY = []
        
        q = self.a
        while q <= 2*self.b - self.a:
            self.realX.append(q)
            self.realY.append(self.func(q))
            q += (self.b - self.a) / 20
        self.predY = self.realY[:self.p] + [0.0] * (len(self.realY) - self.p)
        
    def net(self, w, x):
        net = w[0] + sum(w[i + 1] * x[i] for i in range(self.p))
        return net

    def func(self, t):
        return math.sin(0.1*t**3 - 0.2*t**2 + t-1)
    
    def prediction(self):
        x = self.realY[:self.N] + self.predY[self.N:]
        for i in range(self.N, self.N * 2):
            net = self.w[0]
            for j in range(1, self.p + 1):
                net += self.w[j] * x[i - self.p + j - 1]
            x[i] = net
            self.predY[i] = net

    def learning(self, M = 0):
        if M == 0: M = self.M
        
        x = []
        for i in range(len(self.realX) - self.p):
            x.append(self.realY[i:self.p + i])
        k = 0
        while k < M:
            for i in range(len(x)):
                out = self.net(self.w, x[i])
                self.predY[self.p + i] = out
                delta = self.realY[i + self.p] - out
                
                for j in range(len(self.w) - 1):
                    self.w[j + 1] += self.learnK * delta * x[i][j]
                self.w[0] += self.learnK * delta
            
            self.eps = math.sqrt(sum((self.predY[i - self.p] - self.realY[i])**2 for i in range(self.p, len(self.realY))))
            k += 1
        self.E.append(self.eps)
        self.prediction()
        print('P = {}, n = {:.2}, eras = {}, epsilon = {:.6}, w = '.format(self.p, self.learnK, k, self.eps), end='')
        for i in range(len(self.w) - 1):
            print('{:.4}'.format(self.w[i]), end=', ')
        print('{:.4}'.format(self.w[-1]))

    def results(self):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.plot(self.realX, self.realY)
        plt.show()
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.plot(self.realX, self.realY)
        plt.plot(self.realX, self.predY)
        plt.show()
        
        eps = []
        eras = []
        for k in range(1, 20):
            obj1 = NeuronalNetwork(0, 1, 3000, 0.3)
            obj1.learning(250 * k)
            eps.append(obj1.eps)
            eras.append(k)
        plt.xlabel('M')
        plt.ylabel('eps')
        plt.grid()
        plt.plot(eras, eps)
        plt.show()
        
        eps = []
        learnN = []
        for k in range(1, 21):
            obj2 = NeuronalNetwork(0, 1, 5000, 0.05 * k)
            obj2.learning()
            eps.append(obj2.eps)
            learnN.append(0.05 * k)
        plt.xlabel('n')
        plt.ylabel('eps')
        plt.grid()
        plt.plot(learnN, eps)
        plt.show()
        
        eps = []
        P = []
        for p in range(2, 20):
            obj3 = NeuronalNetwork(0, 1, 5000, 0.3, p) 
            obj3.learning()
            eps.append(obj3.eps)
            P.append(p)
        plt.xlabel('p')
        plt.ylabel('eps')
        plt.grid()
        plt.plot(P, eps)
        plt.show()

obj = NeuronalNetwork(0, 1, 3000, 0.3)
obj.learning()
obj.results()