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

bpn = BackPropagationNetwork((2,2,1))
print(bpn.shape)
print(bpn.weights)
