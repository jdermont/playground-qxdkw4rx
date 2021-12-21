# Introduction

XOR example is a toy problem, a hello world for introducing neural networks. It means you have to build and train the neural network so that given 2 inputs it will output what a XOR function would output (at least close to it). This isn't math heavy explanatory tutorial. You should have at least a vague idea how do neural network work. This article is intended to provide building blocks in form of simple python scripts. No libraries, no numpy are used to build this simple neural network. Beware the style of the python scripts is hackatonish, but I hope more easily understood this way.

# First script

This is simple script, an implementation of *this image*. 

Here the neural network is bunch of loosely written variables. It is trained on xor examples for 10000 epochs, using stochastic gradient descent (or minibatch of size 1 if you like), so no matrix transpositions are needed. Learning rate is 0.1.

The 
```python runnable
import random
import math

w1 = random.uniform(-0.5,0.5)
w2 = random.uniform(-0.5,0.5)
b1 = random.uniform(-0.5,0.5)

w3 = random.uniform(-0.5,0.5)
w4 = random.uniform(-0.5,0.5)
b2 = random.uniform(-0.5,0.5)

w5 = random.uniform(-0.5,0.5)
w6 = random.uniform(-0.5,0.5)
b3 = random.uniform(-0.5,0.5)

o1 = random.uniform(-0.5,0.5)
o2 = random.uniform(-0.5,0.5)
o3 = random.uniform(-0.5,0.5)
ob = random.uniform(-0.5,0.5)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_prime(x): # x already sigmoided
    return x * (1 - x)


def predict(i1,i2):    
    s1 = w1 * i1 + w2 * i2 + b1
    s1 = sigmoid(s1)
    s2 = w3 * i1 + w4 * i2 + b2
    s2 = sigmoid(s2)
    s3 = w5 * i1 + w6 * i2 + b3
    s3 = sigmoid(s3)
    
    output = s1 * o1 + s2 * o2 + s3 * o3 + ob
    output = sigmoid(output)
    
    return output


def learn(i1,i2,target):
    global w1,w2,b1,w3,w4,b2,w5,w6,b3
    global o1,o2,o3,ob
    alpha = 0.1
    
    s1 = w1 * i1 + w2 * i2 + b1
    s1 = sigmoid(s1)
    s2 = w3 * i1 + w4 * i2 + b2
    s2 = sigmoid(s2)
    s3 = w5 * i1 + w6 * i2 + b3
    s3 = sigmoid(s3)
    
    output = s1 * o1 + s2 * o2 + s3 * o3 + ob
    output = sigmoid(output)
    
    error = target - output
    derror = error * sigmoid_prime(output)
    
    ds1 = derror * sigmoid_prime(s1)
    ds2 = derror * sigmoid_prime(s2)
    ds3 = derror * sigmoid_prime(s3)
    
    o1 += alpha * s1 * derror
    o2 += alpha * s2 * derror
    o3 += alpha * s3 * derror
    ob += alpha * derror
    
    w1 += alpha * i1 * ds1
    w2 += alpha * i2 * ds1
    b1 += alpha * ds1
    w3 += alpha * i1 * ds2
    w4 += alpha * i2 * ds2
    b2 += alpha * ds2
    w5 += alpha * i1 * ds3
    w6 += alpha * i2 * ds3
    b3 += alpha * ds3   


INPUTS = [
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ]

OUTPUTS = [
        [0],
        [1],
        [1],
        [0]
    ]


for i in range(10000):
    indexes = [0,1,2,3]
    random.shuffle(indexes)
    for j in indexes:
        learn(INPUTS[j][0],INPUTS[j][1],OUTPUTS[j][0])
    
    if (i+1) % 1000 == 0:
        cost = 0
        for j in range(4):
            o = predict(INPUTS[j][0],INPUTS[j][1])
            cost += (OUTPUTS[j][0] - o) ** 2
        cost /= 4
        print(i+1, "mean squared error:", cost)       
        

print(0,0,predict(0,0))
print(1,0,predict(1,0))
print(0,1,predict(0,1))
print(1,1,predict(1,1))
```

Example output:

```
1000 mean squared error: 0.24979266353990032
2000 mean squared error: 0.24831882619126208
3000 mean squared error: 0.23561863285516624
4000 mean squared error: 0.1780693775264198
5000 mean squared error: 0.06912242900384753
6000 mean squared error: 0.029067840008850473
7000 mean squared error: 0.01615164457711759
8000 mean squared error: 0.01062363347939824
9000 mean squared error: 0.007720927162456013
10000 mean squared error: 0.005980352776240471
0 0 0.08988830233230768
1 0 0.9260414851726995
0 1 0.9254344628052803
1 1 0.06936586304646092
```

Your mileage may vary. Sometimes this simple net will diverge and output for all inputs the 0.666..., or it would need more iterations to train. It's normal as it is more sensitive to starting random weights than more complex models. NN libraries suffer from that too, but they can mitigate it by smarter weights initialization.


# Advanced usage

If you want a more complex example (external libraries, viewers...), use the [Advanced Python template](https://tech.io/select-repo/429)
