from pygame import *
from math import *
from datetime import datetime
from time import time

class Map:
    def __init__(self, width: int, height: int, startPreys: int, startPreds: int, maxPreys: int, maxPreds: int):
        """Creates a map object to hold cells"""
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
        """Signals program ending and ensures accompanying files are also closed"""
        self.file.close()

    def update(self):
        """Updates all cells in the map for the current frame"""
        pass

    def draw(self, screen):
        """Draws all cells in the map for the current frame"""
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

    def turn(self):
        """ Modifies Cell angle by Cell angularVelocity """
        self.angle += self.angularVelocity
        self.angle %= 2 * pi

    def move(self):
        """ Modifies position according to speed, angle and collisionModifier """
        self.xyPos[0] += self.speed * cos(self.angle) + self.collisionModifier[0]
        self.xyPos[1] += self.speed * sin(self.angle) + self.collisionModifier[1]
        self.collisionModifier = [0,0]
    
    def findCollision(self, otherCell):
        """ Detect if collision is happening and modify collisionModifier.

        We check how close this cell and otherCell are to each other. If they are colliding,
        apply a movement on both cells directly away from each other that will be processed
        in `move()`.
        
        """
        v = [self.xyPos[0] - otherCell.xyPos[0], self.xyPos[1] - otherCell.xypos[1]]
        distance = (v[0]**2 + v[1]**2)/2

        if distance > Cell.CELL_RADIUS * 2:
            return

        if distance == 0:
            self.collisionModifier[1] += Cell.MAXIMUM_SPEED
            otherCell.collisionModifier[1] -= Cell.MAXIMUM_SPEED
        
        proximityFactor = 1 - distance/(Cell.CELL_RADIUS * 2)
        v = [v[0]/distance * proximityFactor * Cell.MAXIMUM_SPEED, self.xyPos[1] - v[0]/distance * proximityFactor * Cell.MAXIMUM_SPEED]
        self.collisionModifier[0] -= v[0]
        self.collisionModifier[1] -= v[1]
        otherCell.collisionModifier[0] += v[0]
        otherCell.collisionModifier[1] += v[1]
        return

    def draw(self, canvas):
        """ Draw the cell on `canvas` """
        raise NotImplementedError()

    def canSplit(self):
        """ Virtual method for checking if cell is able to split """
        raise NotImplementedError()

    def split(self):
        if self.canSplit():
            raise NotImplementedError()


class Predator(Cell):
    MAXIMUM_DIGESTION_TIMER = 100
    MAXIMUM_ENERGY = 100
    INITIAL_ENERGY = 50
    PREDATOR_RAY_ANGLES = [-14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14]

    def __init__(self, inheritingNetwork = Cell.EMPTY_NETWORK, previousGenerationNumber = -1, xyPos = [0,0]):
        super().__init__(self, inheritingNetwork, previousGenerationNumber, xyPos)
        self.energy = Predator.INITIAL_ENERGY
        self.digestionTimer = 0

    def getVision(self):
        """ Get vision as input for neural network"""
        raise NotImplementedError()

    def getMove(self):
        """ Feed vision input from `getVision()` into neural network """
        raise NotImplementedError()
    
    def eatPrey(self, victim):
        """ Attempt to eat `victim` """
        raise NotImplementedError()
    
    def canSplit(self):
        """ Check if cell has enough energy to split """
        return self.energy >= Predator.MAXIMUM_ENERGY
