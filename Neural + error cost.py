import math
import random as rand
def NN(m1, m2, w1, w2, b):
  z = m1 * m2 + w1 * w2 + b
  return(sigmoid(z))
def sigmoid(x):
    return 1/(1+math.exp(x))

#to initialise the network, we need to start with random guesses.
w1 = rand.uniform(-4,4)
w2 = rand.uniform(-4,4)
b = rand.uniform(-4,4)
data = [ [3, 1.5, 1],[2, 1, 0], [4, 1.5, 1], [3, 1, 0], [3.5, .5, 1], [2, .5, 0], [5.5,1,1], [1,1,0 ]]
def cost(b):
    return ((b-4)**2)

def slope(b):
    a = 2*(b-4)
    return a

def exampletrainingloop(num):  #essentially this changes our value of b to what we want it to be (4) by removing a portion of the derivive(eqn of slope) and eventually since its a parabola, b --> 4 ( the tp of the parabola.)
    #example of neural network training with no inputs. 
    b =100
    for i in range(num):
        b = b-.1*slope(b)
        print(b)


def test(data):
    for n in range(len(data)):
        dataset = []
        for item in data[n]:

            dataset.append(item)

        foo = dataset[0]
        bar = dataset[1]
        baz = dataset[2]
        print("you said", NN(foo,bar,w1,w2,b ), "but it was", baz)
        print("\n\n")
'''https://www.youtube.com/watch?v=gwitf7ABtK8'''
exampletrainingloop(1000)
