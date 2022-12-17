from pygame import *
from math import *

class Map:
    def __init__(self, width: int, height: int, startPreys: int, startPreds: int, maxPreys: int, maxPreds: int):
        self.width = width
        self.height = height
        self.startPreys = startPreys
        self.startPreds = startPreds
        self.maxPreys = maxPreys
        self.maxPreds = maxPreds

        self.preyList = []
        for i in range(self.startPreys):
            self.preyList.append("prey")

        self.predList = []
        for i in range(self.startPreds):
            self.predList.append("pred")

    def update(self):
        pass

    def draw(self, screen):
        pass

