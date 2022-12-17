
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

class Cell:
    MAXIMUM_SPEED = 100
    MAXIMUM_TURN_SPEED = 0.1
    DEFAULT_ANGLE = 0
    EMPTY_NETWORK = 0
    CELL_RADIUS = 40
    PREY_COLOUR = (0,255,0)
    PREDATOR_COLOUR = (255,0,0)

    def __init__(self, startingNetwork = EMPTY_NETWORK, previousGenerationNumber = -1, xyPos = [0, 0]):
        self.speed = 0
        self.angle = Cell.DEFAULT_ANGLE
        self.angularVelocity = 0
        self.collisionModifier = [0,0]
        self.generationNumber = previousGenerationNumber + 1

    """ Modifies Cell angle by Cell angularVelocity """
    def turn(self):
        self.angle += self.angularVelocity
        self.angle %= 2 * pi

    """ Modifies position according to speed, angle and collisionModifier """
    def move(self):
        self.xyPos[0] += self.speed * cos(self.angle)
        self.xyPos[1] += self.speed * sin(self.angle)

    """ Detect if collision is happening and modify collisionModifier.

    We check how close this cell and otherCell are to each other. If they are colliding,
    apply a movement on both cells directly away from each other that will be processed
    in `move()`.
    
    """
    def findCollision(self, otherCell):
        distance = (self.xyPos[0] - otherCell.xyPos[0])**2 + (self.xyPos[1] - otherCell.xypos[1])**2

        if distance > (Cell.CELL_RADIUS * 2)**2:
            return

        raise NotImplementedError()


    """ Draw the cell on `canvas` """
    def draw(self, canvas):
        raise NotImplementedError()

