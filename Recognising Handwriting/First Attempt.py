import numpy as np

class BackPropagationNetwork:

    layerCount = 0
    shape = None
    weights = []

    def __init__(self, layerSize):
        """initalises the network"""
        #Layer info
        self.layerCount = len(layerSize)-1
        self.shape = layerSize

        #Input /Outut data from last Run
        self._layerInput = []
        self._layerOutput = []

        #Init weight arrays
        for (l1,l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale = 0.1, size = (l2, l1 + 1))) #weight matrix has to have n rows , n columns
    #transfer function
    def sigmoid(self, x, derivative = False):
        if derivative == False:
            return 1/(1+np.exp(-x)) #returns sigmoid(x)
        else:
            out = self.sigmoid(x)
            return out * (1-out) #returns derivitive of sigmoid(x)

    def Run(self, iput):
        """run the network based on input data"""
        #assume input is bunch of rows, where each row is a set of input dataset
        #assuming its a 2d array eg [[a,b,c], [d,e,f]]
        lnCases = iput.shape[0] #number of input InCases

        #Clear out previous intermediate value lists
        self._layerInput =[]
        self._layerOutput = []
        #run it
        #CASE 1(INPUT LAYER)
        for i in range(self.layerCount):
            if i == 0:
                layerInput = self.weights[0].dot(np.vstack([iput.T, np.ones([1, lnCases])])) #self.weights == matrix of weights * matrix of inputs with biases at bottom
                #input.T transposes input from rows to columns so matrix multiplication can be nice
            else:
                layerInput = self.weights[i].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])])) #self.weights == matrix of weights * output of previous layer
            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sigmoid(layerInput))
        return self._layerOutput[-1].T #transposes it backs to rows



bpn = BackPropagationNetwork((2,2,2))
print(bpn.shape)
lvInput = np.array ([[0,0],[1,1],[-1,0.5]])
lvOutput = bpn.Run(lvInput)

print(("Input, \n{0}\n Output:\n {1}").format(lvInput, lvOutput))
