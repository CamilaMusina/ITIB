import matplotlib.pyplot as plt
from itertools import combinations

def drawGraph(E, k, name = 'E(k)'):
    plt.plot(k[1:], E[1:], marker = 'o')
    plt.xlabel('Epoch  k')
    plt.ylabel('Error  E')
    plt.axis([0, k[-1]+1, 0, max(E[1:])+1])
    plt.title('E(k)')
    plt.grid(True)
    plt.savefig('plt_{0}.png'.format(name))
    plt.clf()

x = [0]*16
for i in range(16):
    x[i]=[0]*4
F = []
F = (1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0)
x = ((1,0,0,0,0),
     (1,0,0,0,1),
     (1,0,0,1,0),
     (1,0,0,1,1),
     (1,0,1,0,0),
     (1,0,1,0,1),
     (1,0,1,1,0),
     (1,0,1,1,1),
     (1,1,0,0,0),     
     (1,1,0,0,1),
     (1,1,0,1,0),
     (1,1,0,1,1),
     (1,1,1,0,0),
     (1,1,1,0,1),
     (1,1,1,1,0),
     (1,1,1,1,1))

def net(w,x):
    net = 0
    for i  in range(5):
        net += x[i]*w[i]
return net

def FA(net, fa):
    if fa == 0:
        return 1 if net > 0 else 0
    else:
        return 1 if 0.5*((net/(1.0+abs(net)))+1)>=0.5 else 0

def learn(fa):
    k = 0
    w = [0]*5
    Y = list(FA(net(w,x[i]),0) for i in range(16))
    E = sum((F[i]^Y[i] for i in range(16)))
    arrayK = list(); arrayE = list()
    arrayK.append(k); arrayE.append(E)
    print('F = ', F)
    print('Epoch %d  Y = %s, w = [%.15f, %.15f, %.15f, %.15f, %.15f], E = %d' % (k,str(Y),w[0],w[1],w[2],w[3],w[4],E))
    while E>0:
        k += 1
        arrayK.append(k)
        delta = tuple((F[i]-Y[i] for i in range(16)))
        if fa == 0:
            for i in range(5):
                w[i] += sum(0.3*delta[j]*x[j][i] for j in range(16))
            Y = list(FA(net(w, x[i]),0) for i in range(16))
        else:
            for i in range(5):
                w[i] += sum(0.3*delta[j]*x[j][i] / (1 + abs(net(w, x[i])))**2 for j in range(16))
            Y = list(FA(net(w, x[i]),0) for i in range(16))
        E = sum((F[i]^Y[i] for i in range(16)))
        arrayE.append(E)
        print('Epoch %d  Y = %s, w = [%.15f, %.15f, %.15f, %.15f, %.15f], E = %d' % (k,str(Y),w[0],w[1],w[2],w[3],w[4],E))
    drawGraph(arrayE, arrayK)

def min_learn(sets, flag):
    w = [0] * 5
    Y = list(FA(net(w,x[i]), 2) for i in range(16))
    E = sum((F[i]^Y[i] for i in range(16)))
    arrayK = list(); arrayE = list()
    arrayK.append(0); arrayE.append(E)
    if flag:
        print('Epoch 0 | Y=%s, W=[%.15f, %.15f, %.15f, %.15f, %.15f], E=%d' % (str(Y),w[0],w[1],w[2],w[3],w[4],E))
    k = 1 
    while E>0:
        delta = tuple((F[i]-Y[i] for i in range(16)))
        for i in range(5):
            w[i] += sum(0.3 * delta[sets[j][4]+2*sets[j][3]+4*sets[j][2]+8*sets[j][1]] * sets[j][i] / (1 + abs(net(w, sets[j])))**2 for j in range(sets.__len__()))
        Y = list(FA(net(w,x[i]), 2) for i in range(16))
        E = sum((F[i]^Y[i] for i in range(16)))
        if flag:
            print('Epoch %d | Y=%s, W=[%.15f, %.15f, %.15f, %.15f, %.15f], E=%d' % (k,str(Y),w[0],w[1],w[2],w[3],w[4],E))
        k +=1
        arrayK.append(k)
        arrayE.append(E)
        if k > 64: return -1
    drawGraph(arrayE, arrayK)
    return k-1

def min_set_learn():
    for i in range(2,16):
        comb = list(combinations(x, i)) 
        for sets in comb:
            flag = 1
            count = min_learn(sets, 0)
            if count > 0:
                print('Using ', i, ' vectors')
                print(sets)
                min_learn(sets, 1)
                flag = 0
                break
        if flag == 0:
            break

print('1. FA(1,1):')
learn(0)
print()
print('2. FA(1,2):')
learn(1)
print() 
print('3. Minimum set learning:')
min_set_learn()
