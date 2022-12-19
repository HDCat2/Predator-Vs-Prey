from pygame import *
from math import *
from datetime import datetime
from time import time
from random import *
import numpy as np
import os

class Map:
    HISTORY_INTERVAL = 2

    def __init__(self, width: int, height: int, startPreys: int, startPreds: int, maxPreys: int, maxPreds: int):
        """Creates a map object to hold cells"""
        self.width = width
        self.height = height
        self.startPreys = startPreys
        self.startPreds = startPreds
        self.maxPreys = maxPreys
        self.maxPreds = maxPreds
        self.frameCount = 0
        self.colour = (255, 255, 255)

        self.timer = time()
        self.filename = datetime.now().strftime("Predator-Vs-Prey/Histories/%Y-%m-%d-%H-%M-%S.txt")
        self.file = open(self.filename, "x")

        self.preyList = []
        for i in range(self.startPreys):
            prey = Prey(self, 0, 0, [randint(0, width), randint(0, height)])
            self.preyList.append(prey)

        self.predList = []
        for i in range(self.startPreds):
            pred = Predator(self, 0, 0, [randint(0, width-1), randint(0, height-1)])
            self.predList.append(pred)

    def updateHistory(self):
        if (time() - self.timer) > Map.HISTORY_INTERVAL:
            self.timer = time()
            self.writeInformation()
            #write history into file

    def writeInformation(self):
        self.file.write("Data on frame %d" % self.frameCount)
        self.file.write(datetime.now().strftime(" at %Y/%m/%d %H:%M:%S\n"))
        self.file.write("test\n")
        self.file.write("\n")

    def kill(self):
        """Signals program ending and ensures accompanying files are also closed"""
        self.writeInformation()
        self.file.write("Completed\n")
        self.file.close()

    def update(self):
        """Updates all cells in the map for the current frame"""
        self.updateHistory()

        for prey in self.preyList:
            prey.update()

        for pred in self.predList:
            pred.update()

        self.frameCount += 1

    def draw(self, screen):
        """Draws all cells in the map for the current frame"""
        screen.fill(self.colour)

        for prey in self.preyList:
            prey.draw(screen)

        for pred in self.predList:
            pred.draw(screen)


class Cell:
    MAXIMUM_SPEED = 100
    MAXIMUM_TURN_SPEED = 0.1
    DEFAULT_ANGLE = 0
    EMPTY_NETWORK = 0
    CELL_RADIUS = 10
    CELL_FRONT_LENGTH = 7
    VIEW_DISTANCE = 100
    PREY_COLOUR = (0,255,0)
    PREDATOR_COLOUR = (255,0,0)
    RAY_COLOUR = (210, 210, 210)

    CELL_SETS = None
    BOX_SIZE = None
    BOX_OVERLAP = None
    BOX_HOR_COUNT = 5
    BOX_VER_COUNT = 5

    def __init__(self, map, startingNetwork = EMPTY_NETWORK, previousGenerationNumber = -1, xyPos = None):
        self.speed = 0
        self.angle = Cell.DEFAULT_ANGLE
        self.angularVelocity = 0
        self.collisionModifier = [0,0]
        self.generationNumber = previousGenerationNumber + 1
        self.viewDistance = Cell.VIEW_DISTANCE
        self.lifeLength = 0
        self.type = None #0 for predator, 1 for prey
        self.colour = (0, 0, 0)
        self.xyPos = xyPos
        if self.xyPos == None:
            self.xyPos = [0, 0]
        self.rays = []
        self.rayCount = None
        self.rayGap = None

        if Cell.CELL_SETS == None:
            boxWidth = map.width // Cell.BOX_HOR_COUNT
            boxHeight = map.height // Cell.BOX_VER_COUNT
            Cell.BOX_SIZE = (boxWidth, boxHeight)
            Cell.BOX_OVERLAP = Cell.VIEW_DISTANCE*2 + Cell.CELL_RADIUS*3 + 10
            if Cell.BOX_OVERLAP > max(Cell.BOX_SIZE):
                raise ValueError("Overlap larger than box size, reduce box count")
            Cell.CELL_SETS = [[set() for j in range(Cell.BOX_VER_COUNT)] for i in range(Cell.BOX_HOR_COUNT)]
        setCoord = self.getCenteredSet()
        Cell.CELL_SETS[setCoord[0]][setCoord[1]].add(self)

    def turn(self):
        """ Modifies Cell angle by Cell angularVelocity """
        self.angle += self.angularVelocity
        self.angle %= 2 * pi

    def move(self):
        """ Modifies position according to speed, angle and collisionModifier """
        newPosX = self.xyPos[0] + self.speed * cos(self.angle) + self.collisionModifier[0]
        newPosY = self.xyPos[1] + self.speed * sin(self.angle) + self.collisionModifier[1]
        self.updateSets((newPosX, newPosY))
        self.xyPos[0] = newPosX
        self.xyPos[1] = newPosY
        self.collisionModifier = [0,0]

    def getCollisions(self):
        """ Returns a list of all cells colliding with self """
        setCoord = self.getCenteredSet()
        collisionList = []
        for cell in Cell.CELL_SETS[setCoord[0]][setCoord[1]]:
            if cell != self and self.isColliding(cell):
                collisionList.append(cell)
        return collisionList
    
    def isColliding(self, otherCell):
        """ Detect if collision is happening and modify collisionModifier. """
        v = [self.xyPos[0] - otherCell.xyPos[0], self.xyPos[1] - otherCell.xyPos[1]]
        distance = (v[0]**2 + v[1]**2)**0.5
        if distance > Cell.CELL_RADIUS * 2:
            return False

        #if self.type == otherCell.type:
            #self.repel(otherCell)

        return True

    def kill(self):
        """ Remove the cell from the map """
        for p in self.getSetIndices():
            Cell.CELL_SETS[p[0]][p[1]].remove(self)

    def getSetIndices(self, position = None):
        """ Return set of pairs of indices for which sets the cell belongs to """
        ret = set()
        if position == None:
            position = self.xyPos
        for d in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
            coord = (position[0]+Cell.BOX_OVERLAP*d[0], position[1]+Cell.BOX_OVERLAP*d[1])
            setInd = (int(coord[0]//Cell.BOX_SIZE[0]), int(coord[1]//Cell.BOX_SIZE[1]))
            ret.add(setInd)
        return ret

    def getCenteredSet(self):
        """ Return the indices of the set the cell is most centered in """
        sets = list(self.getSetIndices())
        minDists = []
        for s in sets:
            dists = [abs(self.xyPos[0] - (s[0]*Cell.BOX_SIZE[0] - Cell.BOX_OVERLAP)), abs(self.xyPos[0] - ((s[0]+1)*Cell.BOX_SIZE[0] + Cell.BOX_OVERLAP)),
                     abs(self.xyPos[1] - (s[1]*Cell.BOX_SIZE[1] - Cell.BOX_OVERLAP)), abs(self.xyPos[1] - ((s[1]+1)*Cell.BOX_SIZE[1] + Cell.BOX_OVERLAP))]
            minDists.append(min(dists))
        return sets[minDists.index(max(minDists))]

    def updateSets(self, xyPos2):
        """ Takes new position and updates cell sets to match new position """
        set1 = self.getSetIndices()
        set2 = self.getSetIndices(xyPos2)
        exited = set1 - set2
        entered = set2 - set1
        for coord in exited:
            Cell.CELL_SETS[coord[0]][coord[1]].remove(self)
        for coord in entered:
            Cell.CELL_SETS[coord[0]][coord[1]].add(self)

    def repel(self, otherCell):
        """
        We check how close this cell and otherCell are to each other. If they are colliding,
        apply a movement on both cells directly away from each other that will be processed
        in `move()`.
        """
        v = [self.xyPos[0] - otherCell.xyPos[0], self.xyPos[1] - otherCell.xyPos[1]]
        distance = (v[0] ** 2 + v[1] ** 2) ** 0.5

        if distance == 0:
            self.collisionModifier[1] += Cell.MAXIMUM_SPEED
            otherCell.collisionModifier[1] -= Cell.MAXIMUM_SPEED

        proximityFactor = 1 - distance / (Cell.CELL_RADIUS * 2)
        v = [v[0] / distance * proximityFactor * Cell.MAXIMUM_SPEED,
             self.xyPos[1] - v[0] / distance * proximityFactor * Cell.MAXIMUM_SPEED]
        self.collisionModifier[0] -= v[0]
        self.collisionModifier[1] -= v[1]
        otherCell.collisionModifier[0] += v[0]
        otherCell.collisionModifier[1] += v[1]

    def getIntersectionLength(self, cellAngle, dist, tensor, idx):
        """ Helper function for `getVision()`
        Finds intersection of ray with cell given necessary parameters. Returns True if
        an intersection is found and changes `tensor` at `idx` appropriately, False otherwise.
        """
        rayAngle = abs(cellAngle - (self.angle + self.rays[idx]))
        # Check if ray intersects with circle
        if dist * sin(rayAngle) > Cell.CELL_RADIUS:
            return False

        if rayAngle == cellAngle:
            tensor[idx] = max(tensor[idx], (dist - Cell.CELL_RADIUS)/self.viewDistance)
            return True
        
        # Use cosine law to compute length of ray from origin cell to intersection
        disc = (2 * dist * cos(rayAngle))**2 - 4 * (dist**2 - Cell.CELL_RADIUS**2)
        if disc < 0:
            raise ValueError("Discriminant in getVision() is imaginary!")
        
        root = (2 * dist * cos(rayAngle) - disc**0.5)/2
        if root < 0:
            root += disc**0.5
            if root < 0:
                raise ValueError("Found negative value for length of intersection in getVision()")
        
        tensor[idx] = max(tensor[idx], root/self.viewDistance)
        return True

    def getVision(self):
        """ Get vision as input for neural network """
        #raise NotImplementedError()
        inputTensor = [0 for i in range(self.rayCount)]
        for otherCell in self.getCenteredSet():
            dist = np.linalg.norm(otherCell.xyPos, self.xyPos)

            if dist < Cell.CELL_RADIUS:
                return [1 for i in range(self.rayCount)]
            
            if otherCell.type == 1 or dist >= self.viewDistance + Cell.CELL_RADIUS:
                pass

            cellAngle = atan2(self.xyPos[1] - otherCell.xyPos[1], self.xyPos[0] - otherCell.xyPos[0])

            rayLowerIndex = (cellAngle - self.angle) // self.rayGap + self.rayCount // 2
            rayUpperIndex = rayLowerIndex + 1

            while rayLowerIndex >= 0:
                ret = self.getIntersectionLength(cellAngle, dist, inputTensor, rayLowerIndex)
                if ret:
                    rayLowerIndex -= 1
                else:
                    break
            
            while rayUpperIndex < self.rayCount:
                ret = self.getIntersectionLength(cellAngle, dist, inputTensor, rayUpperIndex)
                if ret:
                    rayUpperIndex += 1
                else:
                    break

    def update(self):
        self.move()

    def draw(self, screen, drawRays = False):
        """ Draw the cell on `canvas` """
        if drawRays:
            for ray in self.rays:
                rayDest = (self.xyPos[0] + self.viewDistance*cos(self.angle + ray), self.xyPos[1] + self.viewDistance*sin(self.angle + ray))
                draw.line(screen, Cell.RAY_COLOUR, self.xyPos, rayDest, 1)
        draw.circle(screen, self.colour, self.xyPos, Cell.CELL_RADIUS, 0)

        #draw.line(screen, self.colour, self.xyPos, Cell.CELL_FRONT_LENGTH, 2)
        #draw outward rays

class Predator(Cell):
    MAXIMUM_DIGESTION_TIMER = 100
    MAXIMUM_ENERGY = 100
    INITIAL_ENERGY = 50
    RAY_COUNT = 15
    RAY_GAP = 0.06
    VIEW_DISTANCE = 100

    def __init__(self, map, inheritingNetwork = Cell.EMPTY_NETWORK, previousGenerationNumber = -1, xyPos = None):
        if xyPos == None:
            xyPos = [0, 0]
        super().__init__(map, inheritingNetwork, previousGenerationNumber, xyPos)
        self.energy = Predator.INITIAL_ENERGY
        self.digestionTimer = 0
        self.type = 0
        self.colour = Cell.PREDATOR_COLOUR
        self.rays = [-Predator.RAY_GAP*Predator.RAY_COUNT//2 + Predator.RAY_GAP*i for i in range(Predator.RAY_COUNT)]
        self.rayCount = Predator.RAY_COUNT
        self.rayGap = Predator.RAY_GAP
        self.viewDistance = Predator.VIEW_DISTANCE

    def getMove(self):
        """ Feed vision input from `getVision()` into neural network """
        raise NotImplementedError()
    
    def eatPrey(self, victim):
        """ Attempt to eat `victim` """
        raise NotImplementedError()
    
    def canSplit(self):
        """ Check if cell has enough energy to split """
        return self.energy >= Predator.MAXIMUM_ENERGY

    def split(self):
        if self.canSplit:
            raise NotImplementedError()

class Prey(Cell):
    MAXIMUM_ENERGY = 100
    INITIAL_ENERGY = 50
    LIFESPAN = 10
    RAY_COUNT = 15
    RAY_GAP = 0.2
    VIEW_DISTANCE = 50

    def __init__(self, map, inheritingNetwork = Cell.EMPTY_NETWORK, previousGenerationNumber = -1, xyPos = None):
        if xyPos == None:
            xyPos = [0, 0]
        super().__init__(map, inheritingNetwork, previousGenerationNumber, xyPos)
        self.energy = Prey.INITIAL_ENERGY
        self.type = 1
        self.colour = Cell.PREY_COLOUR
        self.rays = [-Prey.RAY_GAP*Prey.RAY_COUNT//2 + Prey.RAY_GAP*i for i in range(Prey.RAY_COUNT)]
        self.rayCount = Prey.RAY_COUNT
        self.rayGap = Prey.RAY_GAP
        self.viewDistance = Prey.VIEW_DISTANCE

    def getMove(self):
        """ Feed vision input from `getVision()` into neural network """
        raise NotImplementedError()

    def canSplit(self):
        """ Check if cell has lived long enough to split """
        return self.lifeLength > Prey.LIFESPAN

    def split(self):
        if self.canSplit:
            raise NotImplementedError()