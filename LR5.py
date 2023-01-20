import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

def get_net_forward(signal, weight):
    return np.dot(signal, weight)

def get_net_backward(response, weight):
    return np.dot(response, weight.transpose())

def activation_function(net):
    result = [[-1 if element <= 0 else 1 for element in row] for row in net]
    return np.array(result)
    
def activation_function(net, prev_f):
    for row in range(len(net)):
        for idx in range(len(net[row])):
            if net[row, idx] == 0:
                net[row, idx] = prev_f[row, idx]
            elif net[row, idx] < 0:
                net[row, idx] = -1
            else:
                net[row, idx] = 1
    return net

def show_signal(signal):
    signal_binary = np.array([[0 if element == -1 else 1 for element in row] for row in signal])
    plt.imsave('filename_1.png', np.array(signal_binary[0]).reshape(6,4), cmap=cm.gray)
    plt.imsave('filename_2.png', np.array(signal_binary[1]).reshape(6,4), cmap=cm.gray)
    plt.imsave('filename_3.png', np.array(signal_binary[2]).reshape(6,4), cmap=cm.gray)
    img_1 = mpimg.imread('filename_1.png')
    img_2 = mpimg.imread('filename_2.png')
    img_3 = mpimg.imread('filename_3.png')
    plt.subplot(131)
    plt.imshow(img_1)
    plt.subplot(132)
    plt.imshow(img_2)
    plt.subplot(133)
    plt.imshow(img_3)
    plt.show()

def get_response(signal, weight, expected_signal):
    net = get_net_forward(signal, weight)
    response_got = activation_function(net, expected_signal)
    return response_got

def get_signal(response, weight, expected_response):
    net = get_net_backward(response, weight)
    signal_got = activation_function(net, expected_response)
    return signal_got

if __name__ == '__main__':
    response = np.array([[-1, -1, 1], [-1, 1, -1], [-1, 1, 1]])

    signal = np.array([
        [1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1], 
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1], 
        [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1]]
        )
    
    show_signal(signal)
    
    weight = np.dot(signal.transpose(), response)
    print(weight.shape)
    
    response_got = get_response(signal, weight, signal)
    if (response_got == response).all():
        print("\nNeurons of second layer:\n", response_got)
        
    signal_got = get_signal(response_got, weight, response)
    if (signal_got == signal).all():
        print("\nNeurons of first layer.\n")
        show_signal(signal_got)
        
    noisy_signal = signal.copy()
    for i in range(noisy_signal.shape[0]):
        np.random.seed = 24
        idx_to_change = np.random.randint(0, noisy_signal.shape[1], 3)
        for idx in idx_to_change:
            noisy_signal[i, idx] *= (-1)
            
    print("\nNeurons with input vector distortion:\n", noisy_signal)
    show_signal(noisy_signal)
    
    response_got = get_response(noisy_signal, weight, signal)
    signal_got = get_signal(response_got, weight, response)
    show_signal(signal_got)