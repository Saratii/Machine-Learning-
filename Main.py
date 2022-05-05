import random
import numpy as np
from multiprocessing import Pool
import cProfile
from copy import deepcopy
squares = []
notSquares = []
bestNet = 0
GRID_SIZE = 11
NET_COUNT = 100

def buildDataSet():
    for _ in range(500): 
        square = np.array([[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] ,dtype = 'f')
        x = random.randint(0, GRID_SIZE-1) 
        y = random.randint(0, GRID_SIZE-1) 
        side = random.randint(1, min(GRID_SIZE-x, GRID_SIZE-y)) 
        for j in range(side): 
            for k in range(side): 
                square[x+j][y+k] = 1 
        squares.append(np.array([[item for sublist in square for item in sublist]], dtype = 'f')) 
        notSquare = square
        x = random.randint(0, GRID_SIZE-1) 
        y = random.randint(0 , GRID_SIZE-1)
        notSquare[x][y] = 1-notSquare[x][y] 
        notSquares.append(np.array([[item for sublist in notSquare for item in sublist]], dtype = 'f'))  
buildDataSet()

class neuralNet:
    def __init__(self, matrix1, matrix2, matrix3): 
        self.matrix1 = matrix1
        self.matrix2 = matrix2
        self.matrix3 = matrix3
    def normalize(self, /, arr: list): 
        k = np.linalg.norm(arr)
        if k == 0:
            return arr
        return arr/k
    def classify(self, data):
        data = np.matmul(data, self.matrix1)
        self.normalize(data)
        data = np.matmul(data, self.matrix2)
        self.normalize(data)
        data = np.matmul(data, self.matrix3)
        self.normalize(data)
        return data[0][0] > data[0][1] 
    def score(self):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for square in squares:
            if self.classify(square):
                tp += 1
            else:
                fn += 1
        for notSquare in notSquares:
            if not self.classify(notSquare):
                tn += 1
            else: 
                fp += 1
        return (tp + tn) / (tp + fp + tn + fn)
    def __deepcopy__(self, _):
        copy1 = self.matrix1.copy()
        copy2 = self.matrix2.copy()
        copy3 = self.matrix3.copy()
        return neuralNet(copy1, copy2, copy3)

def mootate(net): 
    height, width = net.matrix1.shape
    i = random.randint(0, height-1)
    j = random.randint(0, width-1) 
    net.matrix1[i][j] = max(min(net.matrix1[i][j] + random.randint(-25, 25)/50, 1), -1) 
    height, width = net.matrix2.shape
    i = random.randint(0, height-1)
    j = random.randint(0, width-1) 
    net.matrix2[i][j] = max(min(net.matrix2[i][j] + random.randint(-25, 25)/50, 1), -1)
    height, width = net.matrix3.shape
    i = random.randint(0, height-1)
    j = random.randint(0, width-1) 
    net.matrix3[i][j] = max(min(net.matrix3[i][j] + random.randint(-25, 25)/50, 1), -1)
    return neuralNet(net.matrix1, net.matrix2, net.matrix3)      
def scoreNets(net):
    return (net, net.score())
def mySort(scoredNet1): 
    _,score1 = scoredNet1
    return score1
def test():
    test = [[ 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]]
    
    print(bestNet.classify(np.array(test))) 
def train():
    nets = []
    generation = 0 
    scoredNets = []
    for i in range(NET_COUNT): 
        nets.append(neuralNet(
            np.array([[0 for _ in range(80)] for _ in range(121)], dtype = "f"),
            np.array([[0 for _ in range(40)] for _ in range(80 )], dtype = "f"),
            np.array([[0 for _ in range(2 )] for _ in range(40 )], dtype = "f")
        ))

    while True:
        highest_score = -10
        with Pool(8) as p:
            scoredNets = p.map(scoreNets, nets)
        for net, score in scoredNets:
            if score > highest_score:
                highest_score = score
                global bestNet
                bestNet = deepcopy(net)
        scoredNets.sort(reverse = True, key = mySort)
        for i in range(int(NET_COUNT/2)): 
            net,_ = scoredNets[i] 
            nets[i] = mootate(net) 
            nets[int(NET_COUNT/2)+i] = mootate(net) 
        generation += 1
        print(f'Generation: {generation} Score {highest_score}!') 
        if generation == 2000 or highest_score == len(squares) + len(notSquares): 
            break
    
    print(f'Best Net: {bestNet.matrix1}', "\n\n", bestNet.matrix2, "\n\n", bestNet.matrix3)
 
    
if __name__ == '__main__':
    cProfile.run('train()')
    train()
    test()
