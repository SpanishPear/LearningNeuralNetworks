import math
import random as rand

#COST = SIGMA FROM i = 1 to i = m [model(x1)-target(i)]**2
#averageCost = 1/m * COST

#to initialise the network, we need to start with random guesses.

data = [ [3, 1.5, 1],[2, 1, 0], [4, 1.5, 1], [3, 1, 0], [3.5, .5, 1], [2, .5, 0], [5.5,1,1], [1,1,0 ]]
#[4.5,1,"should be 1"]


def NN(m1, m2, w1, w2, b):
  z = m1 * m2 + w1 * w2 + b #weighing * length + weighting * width + global output constant
  return(sigmoid(z))
def sigmoid(x):
    return 1/(1+math.exp(x))

def cost(b):
    return ((b-4)**2)

def slope(m1,m2,w1,w2,b):
    a = 2*(b-4)
    return a


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
#test(data)


def train():
    w1 = rand.uniform(-4,4)
    w2 = rand.uniform(-4,4)
    b = rand.uniform(-4,4)
    rate = 0.2
    for i in range(100000):
        ## picks a random flower and gets what it "should be", the third (second) value in the array
        numflowers = len(data)
        random_flower = rand.randint(0,numflowers-1)
        flower = data[random_flower]
        target = flower[2]

        #feed it to the neural network to get a guess between 0 and 1
        z = w1 * flower[0] + w2 * flower[1] + b
        prediction = sigmoid(z)

        #compare model prediction to target using cost forula (prediction -target) squared
        cost = (prediction -target)**2

        #find slope of cost with relation two each parameter (w1, w2, b) ie differentiate in terms of prediction(?)
        cost_slope = 2* (prediction -target)

        #find deriviitives of z in terms of w1, w2, b and derivitive of prediction in terms of sigmoid ( some advanced maths that i havent learnt yet )
        dpredicition_dz = sigmoid(z) * (1-sigmoid(z))
        dz_dw1= flower[0]
        dz_dw2 = flower[2]
        dz_db = 1

        #find deriviitives of cost in terms of w1, w2 and b
        #d/d2(cost)=2(prediction-target)* d/dz(prediction) * d/dw(target)

        dcost_dw1 = cost_slope * dpredicition_dz * dz_dw1
        dcost_dw2 = cost_slope * dpredicition_dz * dz_dw2
        dcost_db = cost_slope * dpredicition_dz * dz_db

        #update based on stuff we did
        w1 -= rate* dcost_dw1
        w2 -= rate * dcost_dw2
        b -= rate * dcost_db
    print(sigmoid(w1 * 4.5 + w2 * 1 + b)) #( should be ~~ 1)
train()
