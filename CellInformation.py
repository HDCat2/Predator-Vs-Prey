
from math import *
from datetime import datetime
from time import time

class Map:
    def __init__(self, width: int, height: int, startPreys: int, startPreds: int, maxPreys: int, maxPreds: int):
        '''Creates a map object to hold cells'''
        self.width = width
        self.height = height
        self.startPreys = startPreys
        self.startPreds = startPreds
        self.maxPreys = maxPreys
        self.maxPreds = maxPreds

        self.filename = datetime.now().strftime("Histories/%Y%m%d%H%M%S.txt")
        self.timer = time()

        self.file = open(self.filename, "w")

        self.preyList = []
        for i in range(self.startPreys):
            self.preyList.append("prey")

        self.predList = []
        for i in range(self.startPreds):
            self.predList.append("pred")

    def updateHistory(self):
        if (self.timer - time()) > 1000:
            self.timer = time()
            self.file.write("test")
            #write history into file

    def kill(self):
        '''Signals program ending and ensures accompanying files are also closed'''
        self.file.close()

    def update(self):
        '''Updates all cells in the map for the current frame'''
        pass

    def draw(self, screen):
        '''Draws all cells in the map for the current frame'''
        pass

m = Map(1, 1, 1, 1, 1, 1)
while True:
    m.updateHistory()
