import numpy as np
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm

def loss_func(y_true, y_predicted):
    return y_true - y_predicted

def MSE(y_true, y_predicted):
    return np.square(np.sum((y_true - y_predicted) ** 2))

class ActivationFunction:
    def forward(self, x):
        self.x = x
        self.out = (1 - np.exp(-x)) / (1 + np.exp(-x))
        return self.out
    
    def backward(self, delta, lr=1):
        return delta * 0.5 * (1 - self.out ** 2)

class Dense:
    def __init__(self, in_size, out_size, seed=0):
        np.random.seed(seed)        
        self.W = np.random.normal(scale=0.1, size=(out_size, in_size))
        self.b = np.random.normal(scale=0.1, size=(out_size))
        
    def forward(self, x):
        self.x = x
        self.net = np.dot(self.W, x.transpose()) + self.b
        return self.net
    
    def backward(self, delta, lr=1):
        self.dW = np.outer(delta, self.x)
        self.db = delta
        
        self.next_delta = np.dot(delta, self.W) 
        
        self.W = self.W + lr * self.dW
        self.b = self.b + lr * self.db
        
        return self.next_delta
    
class FullyConnectedNeuralNetwork:
    
    def __init__(self):
        self.d1 = Dense(1, 2)
        self.a1 = ActivationFunction()
        self.d2 = Dense(2, 1)
        self.a2 = ActivationFunction()
        
    def forward(self, x):
        net = self.d1.forward(x)
        net = self.a1.forward(net)
        net = self.d2.forward(net)
        net = self.a2.forward(net)
        
        self.net = net
        return net
    
    def backward(self, dz, lr):
        dz = self.a2.backward(dz, lr)
        dz = self.d2.backward(dz, lr)
        dz = self.a1.backward(dz, lr)
        dz = self.d1.backward(dz, lr)
        return dz
    
    
    
    
net = FullyConnectedNeuralNetwork()
X = np.array([-3])
Y = np.array([-1 / 10])
X_train = X
X_test = X
lr = 1
epsilon = 1e-20
loss_train = []
loss_test = []
loss_mse = []

for i in tqdm(range(100)):
    y_predicted = net.forward(X_train)
    delta = loss_func(Y, y_predicted)
    net.backward(delta, lr)
    loss_train.append(delta.item())
    
    y_predicted = net.forward(X_test)
    delta = loss_func(Y, y_predicted)
    loss_test.append(delta.item())
    
    mse = MSE(Y, y_predicted)
    loss_mse.append(mse)
    
    print(f"epoch : {i}\t predict : {y_predicted[0]:.3f}\t MSE : {mse:.3}\t error : {delta[0]:.3}")
    if mse <= epsilon:
        break
    
plt.plot(loss_train)
plt.grid()
plt.title('Dependence between error and epoch number on train')
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()

plt.plot(loss_test)
plt.grid()
plt.title('Dependence between error and epoch number on test')
plt.xlabel('epoch')
plt.ylabel('error')
plt.show()

plt.plot(loss_mse)
plt.grid()
plt.title('Dependence between MSE and epoch number')
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.show()

print(net.d1.W, net.d1.b)
print(net.d2.W, net.d2.b)