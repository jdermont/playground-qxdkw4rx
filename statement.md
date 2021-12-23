# Introduction

XOR example is a toy problem in machine learning community, a hello world for introducing neural networks. It means you have to build and train the neural network so that given 2 inputs it will output what a XOR function would output (at least close to it). This isn't math heavy explanatory tutorial, there are plenty of them out there. I assume you have some vague knowledge of neural networks and try to write a simple one. This article is just a bunch of simple python scripts that implement neural networks. No numpy or other libraries are used, so they should be easily translatable to other languages.

All the scripts use stochastic gradient descent to train the neural network, one data row at a time, so no need for matrix tranpositions. The loss function is mean squared error.

# First script

This is the simplest script, an implementation of *this image*. 

Here the neural network is just a bunch of loosely written variables.

```python runnable
import random
import math

VARIANCE_W = 0.5
VARIANCE_B = 0.1
w1 = random.uniform(-VARIANCE_W,VARIANCE_W)
w2 = random.uniform(-VARIANCE_W,VARIANCE_W)
b1 = random.uniform(-VARIANCE_B,VARIANCE_B)

w3 = random.uniform(-VARIANCE_W,VARIANCE_W)
w4 = random.uniform(-VARIANCE_W,VARIANCE_W)
b2 = random.uniform(-VARIANCE_B,VARIANCE_B)

w5 = random.uniform(-VARIANCE_W,VARIANCE_W)
w6 = random.uniform(-VARIANCE_W,VARIANCE_W)
b3 = random.uniform(-VARIANCE_B,VARIANCE_B)

o1 = random.uniform(-VARIANCE_W,VARIANCE_W)
o2 = random.uniform(-VARIANCE_W,VARIANCE_W)
o3 = random.uniform(-VARIANCE_W,VARIANCE_W)
ob = random.uniform(-VARIANCE_B,VARIANCE_B)


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


def learn(i1,i2,target, alpha=0.2):
    global w1,w2,b1,w3,w4,b2,w5,w6,b3
    global o1,o2,o3,ob
    
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
    
    ds1 = derror * o1 * sigmoid_prime(s1)
    ds2 = derror * o2 * sigmoid_prime(s2)
    ds3 = derror * o3 * sigmoid_prime(s3)
    
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


for epoch in range(1,10001):
    indexes = [0,1,2,3]
    random.shuffle(indexes)
    for j in indexes:
        learn(INPUTS[j][0],INPUTS[j][1],OUTPUTS[j][0], alpha=0.2)
    
    if epoch%1000 == 0:
        cost = 0
        for j in range(4):
            o = predict(INPUTS[j][0],INPUTS[j][1])
            cost += (OUTPUTS[j][0] - o) ** 2
        cost /= 4
        print("epoch", epoch, "mean squared error:", cost)       
        

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

Your mileage may vary. Sometimes this simple net will diverge and output for all inputs the 0.666..., or it would need more iterations to train. It's normal as it is more sensitive to starting random weights than more complex models. NN libraries suffer from that too, but they can mitigate it by smarter weights initialization. You can play around with the learning rate (alpha) or the random bounds (VARIANCE_W, VARIANCE_B).


# Advanced usage

If you want a more complex example (external libraries, viewers...), use the [Advanced Python template](https://tech.io/select-repo/429)
