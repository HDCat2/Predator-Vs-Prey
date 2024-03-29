import pygame as pyg
import math
from datetime import datetime
from time import time
import random as rdm
import numpy as np
import CellUtil as cu
import CellAI as ca
import os
from torch import cuda
from copy import deepcopy

class Map:
    HISTORY_INTERVAL = 100
    MUTATE_INTERVAL = 2

    def __init__(self, width: int, height: int, startPreys: int, startPreds: int, maxPreys: int, maxPreds: int, doLogging: bool):
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
        self.doLogs = doLogging

        if self.doLogs:
            self.filename = datetime.now().strftime(__file__ + "\..\Histories\%Y-%m-%d-%H-%M-%S.txt")
            self.file = open(self.filename, "x")

        self.preyList = []
        for i in range(self.startPreys):
            prey = Prey(self, 0, 0, [rdm.randint(0, width-1), rdm.randint(0, height-1)])
            self.preyList.append(prey)

        self.predList = []
        for i in range(self.startPreds):
            pred = Predator(self, 0, 0, [rdm.randint(0, width-1), rdm.randint(0, height-1)])
            self.predList.append(pred)

    def updateHistory(self):
        if self.doLogs:
            self.writeInformation()
            #write history into file

    def writeInformation(self):
        if not self.doLogs:
            raise AssertionError("Logging is turned off in this instance!")
        
        self.file.write("Data on frame %d" % self.frameCount)
        self.file.write(datetime.now().strftime(" at %Y/%m/%d %H:%M:%S\n"))
        self.file.write("test\n")
        self.file.write("\n")

    def kill(self):
        """Signals program ending and ensures accompanying files are also closed"""
        if self.doLogs:
            self.writeInformation()
            self.file.write("Completed\n")
            self.file.close()
        Cell.CELL_SETS = None

    def dupCell(self, cell):
        """Takes a cell and duplicates it in the map"""
        type = cell.type

        if type == 0:
            newCell = Predator(self, deepcopy(cell.startingNetwork), cell.generationNumber + 1, cell.xyPos)
            self.predList.append(newCell)
        else:
            newCell = Prey(self, deepcopy(cell.startingNetwork), cell.generationNumber + 1, cell.xyPos)
            self.preyList.append(newCell)

    def update(self):
        """Updates all cells in the map for the current frame"""

        # Splitting
        for prey in self.preyList:
            if prey.canSplit():
                prey.split()

        for pred in self.predList:
            if pred.canSplit():
                pred.split()

        # Movement
        for prey in self.preyList:
            prey.update()

        for pred in self.predList:
            pred.update()

        # Deal with collisions
        for prey in self.preyList:
            for cell in prey.getCollisions():
                prey.repel(cell)

        for pred in self.predList:
            for cell in pred.getCollisions():
                if cell.type == 1:
                    if not pred.eatPrey(cell):
                        pred.repel(cell)
                else:
                    pred.repel(cell)

        # Killing dead cells
        preyList = self.preyList[:]
        self.preyList = []
        for prey in preyList:
            if prey.alive:
                self.preyList.append(prey)

        predList = self.predList[:]
        self.predList = []
        for pred in predList:
            if pred.alive:
                self.predList.append(pred)

        # Mutations
        if self.frameCount % Map.HISTORY_INTERVAL == 0:
            self.timer = time()
            self.updateHistory()

            for prey in self.preyList:
                prey.mutate()

            for pred in self.predList:
                pred.mutate()

        self.frameCount += 1

    def draw(self, screen):
        """Draws all cells in the map for the current frame"""
        screen.fill(self.colour)

        for prey in self.preyList:
            prey.draw(screen, False)

        for pred in self.predList:
            pred.draw(screen, False)


class Cell:
    MAXIMUM_SPEED = 0.5
    MAXIMUM_TURN_SPEED = 0.001
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

    def __init__(self, cellMap, startingNetwork = EMPTY_NETWORK, previousGenerationNumber = -1, xyPos = None):
        self.alive = True
        self.selected = False
        self.speed = 0
        self.angle = Cell.DEFAULT_ANGLE
        self.angle = rdm.random() * math.pi * 2
        self.angularVelocity = 0
        self.collisionModifier = [0,0]
        self.startingNetwork = None
        self.generationNumber = previousGenerationNumber + 1
        self.viewDistance = Cell.VIEW_DISTANCE
        self.type = None #0 for predator, 1 for prey
        self.colour = (0, 0, 0)
        self.xyPos = [float(xyPos[0]), float(xyPos[1])]
        if not self.xyPos:
            self.xyPos = [0.0, 0.0]
        self.rays = []
        self.rayCount = None
        self.rayGap = None
        self.map = cellMap
        self.energy = None
        self.maxEnergy = None
        self.digestionTimer = None
        self.lifeLength = None
        self.tensorVision = []

        if Cell.CELL_SETS == None:
            Cell.BOX_HOR_COUNT = self.map.width//(Cell.CELL_RADIUS * 4 + Cell.VIEW_DISTANCE * 2+5)
            Cell.BOX_VER_COUNT = self.map.height//(Cell.CELL_RADIUS * 4 + Cell.VIEW_DISTANCE * 2+5)
            boxWidth = self.map.width / Cell.BOX_HOR_COUNT
            boxHeight = self.map.height / Cell.BOX_VER_COUNT
            Cell.BOX_SIZE = (boxWidth, boxHeight)
            Cell.BOX_OVERLAP = Cell.VIEW_DISTANCE + Cell.CELL_RADIUS*2 + 1
            if Cell.BOX_OVERLAP >= min(Cell.BOX_SIZE)//2:
                raise ValueError("Overlap larger than box size, reduce box count")
            Cell.CELL_SETS = [[set() for j in range(Cell.BOX_VER_COUNT)] for i in range(Cell.BOX_HOR_COUNT)]

        setCoords = self.getSetIndices()
        for coord in setCoords:
            Cell.CELL_SETS[coord[0]][coord[1]].add(self)

    def turn(self):
        """ Modifies Cell angle by Cell angularVelocity """
        self.angle += self.angularVelocity
        self.angle %= 2 * math.pi

    def move(self):
        """ Modifies position according to speed, angle and collisionModifier """
        newPosX = (self.xyPos[0] + self.speed * math.cos(self.angle) + self.collisionModifier[0]) % self.map.width
        newPosY = (self.xyPos[1] + self.speed * math.sin(self.angle) + self.collisionModifier[1]) % self.map.height
        self.updateSets((newPosX, newPosY))
        self.xyPos[0] = newPosX
        self.xyPos[1] = newPosY
        self.collisionModifier = [0,0]

    def getCollisions(self):
        """ Returns a list of all cells colliding with self """
        setCoord = self.getSetIndex()
        collisionList = []
        for cell in Cell.CELL_SETS[setCoord[0]][setCoord[1]]:
            if cell.alive and cell != self and self.isColliding(cell):
                collisionList.append(cell)
        return collisionList

    def wrapCoords(self, coords):
        """ Takes a set of coordinates, and if within the same square as self (in wrapped map), returns coords if they were near self """
        x, y = coords
        if x < self.map.width//2:
            if abs(self.xyPos[0] - x) > abs(self.xyPos[0] - (x + self.map.width)):
                x = x + self.map.width
        else:
            if abs(self.xyPos[0] - x) > abs(self.xyPos[0] - (x - self.map.width)):
                x = x - self.map.width
        if y < self.map.height//2:
            if abs(self.xyPos[1] - y) > abs(self.xyPos[1] - (y + self.map.height)):
                y = y + self.map.height
        else:
            if abs(self.xyPos[1] - y) > abs(self.xyPos[1] - (y - self.map.height)):
                y = y - self.map.height
        return (x, y)
    
    def isColliding(self, otherCell):
        """ Detect if collision is happening and modify collisionModifier.
        This method should only be called once per pair of cells for each frame.
        """
        wrappedCoords = self.wrapCoords(otherCell.xyPos)
        distance = np.linalg.norm(np.subtract(self.xyPos, wrappedCoords))
        if distance > Cell.CELL_RADIUS * 2:
            return False

        #if self.type == otherCell.type:
            #self.repel(otherCell)

        return True

    def kill(self):
        """ Remove the cell from the map """
        self.alive = False
        for p in self.getSetIndices():
            Cell.CELL_SETS[p[0]][p[1]].remove(self)

    def getSetIndices(self, position = None):
        """ Return set of pairs of indices for which sets the cell belongs to """
        ret = set()
        if position == None:
            position = self.xyPos
        for d in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            coord = ((position[0]+Cell.BOX_OVERLAP*d[0]) % self.map.width, (position[1]+Cell.BOX_OVERLAP*d[1]) % self.map.height)
            setInd = self.getSetIndex(coord)
            ret.add(setInd)
        return ret

    def getSetIndex(self, coord = None):
        """ Return the indices of the set the cell is in, not including the overlaps """
        if not coord:
            coord = self.xyPos
        return (int(coord[0]//Cell.BOX_SIZE[0]), int(coord[1]//Cell.BOX_SIZE[1]))

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
        otherCellCoords = self.wrapCoords(otherCell.xyPos)

        v = np.subtract(otherCellCoords, self.xyPos)
        distance = np.linalg.norm(v)

        if distance == 0:
            self.collisionModifier[1] += Cell.MAXIMUM_SPEED
            #therCell.collisionModifier[1] -= Cell.MAXIMUM_SPEED
            return

        proximityFactor = 1 - distance / (Cell.CELL_RADIUS * 2)
        v = [v[0] / distance * proximityFactor * Cell.MAXIMUM_SPEED,
             v[1] / distance * proximityFactor * Cell.MAXIMUM_SPEED]
        
        self.collisionModifier[0] -= v[0]
        self.collisionModifier[1] -= v[1]
        otherCell.collisionModifier[0] += v[0]
        otherCell.collisionModifier[1] += v[1]

    def getIntersectionLength(self, cellAngle, dist, rayIdx, otherCell):
        """ Helper function for `getVision()`
        Finds length of intersection of ray with cell given necessary parameters. Returns
        weight for neural network between 0 and 1 (0 if no intersection)
        Ray assumed to originate from self, and angle is calculated using cellAngle and rayIdx
        """
        otherCellCoords = self.wrapCoords(otherCell.xyPos)

        rayAngle = self.angle + self.rays[rayIdx]
        # Check if ray intersects with circle
        if cu.minDistanceOfRayFromPoint(self.xyPos, [math.cos(rayAngle), math.sin(rayAngle)], otherCellCoords) > Cell.CELL_RADIUS:
            return self.viewDistance


        adjustedRayAngle = abs(cellAngle - rayAngle) # angle in triangle that we will try to compute intersection length with
        if adjustedRayAngle <= 0.000001:
            return dist - Cell.CELL_RADIUS


        # Use cosine law to compute length of ray from origin cell to intersection
        disc = (2 * dist * math.cos(adjustedRayAngle))**2 - 4 * (dist**2 - Cell.CELL_RADIUS**2)

        if disc < 0:
            disc = 0 # It's probably the tangent case - set to 0
        
        root = (2 * dist * math.cos(adjustedRayAngle) - disc**0.5)/2
        if root < 0:
            root = 0 # It's probably an FP error caused by the cells being too close - set to 0


        return root

    def getVisionOfCell(self, otherCell):
        """ Get vision of other cell
        Finds lengths of vision ray intersections originating from self and going to otherCell
        Returns list corresponding to neural net weights
        This is different from getVision as this method behaves as if only self and otherCell exist
        on the map
        """
        outputTensor = [self.VIEW_DISTANCE for i in range(self.rayCount)]
        otherCellCoords = self.wrapCoords(otherCell.xyPos)

        dist = np.linalg.norm(np.subtract(otherCellCoords, self.xyPos))

        # Check if the center of the cell is inside the other cell
        if dist < Cell.CELL_RADIUS:
            return [0 for i in range(self.rayCount)]

        # Check if cell is out of range
        if otherCell.type == self.type or dist >= self.viewDistance + Cell.CELL_RADIUS:
            return outputTensor

        cellAngle = math.atan2(otherCellCoords[1] - self.xyPos[1], otherCellCoords[0] - self.xyPos[0]) % (math.pi*2)

        # Attempt to find 2 adjacent rays that hit `otherCell`
        sign = 1
        if cellAngle < self.angle:
            sign = -1
        rayLowerIndex = int(sign * (abs(cellAngle - self.angle) % (math.pi*2)) // self.rayGap + self.rayCount // 2)
        rayUpperIndex = rayLowerIndex + 1

        if self.selected:
            print(123)
            print(otherCell)
            print(outputTensor)
            print(rayLowerIndex, cellAngle, self.angle)
        
        # Decrease rayLowerIndex until we stop finding rays that intersect
        if 0 <= rayLowerIndex < self.rayCount:
            while rayLowerIndex >= 0:
                outputTensor[rayLowerIndex] = self.getIntersectionLength(cellAngle, dist, rayLowerIndex, otherCell)
                if outputTensor[rayLowerIndex] < self.VIEW_DISTANCE:
                    rayLowerIndex -= 1
                else:
                    break
        
        # Increase rayUpperIndex until we stop finding rays that intersect
        if 0 <= rayUpperIndex < self.rayCount:
            while rayUpperIndex < self.rayCount:
                outputTensor[rayUpperIndex] = self.getIntersectionLength(cellAngle, dist, rayUpperIndex, otherCell)
                if outputTensor[rayUpperIndex] < self.VIEW_DISTANCE:
                    rayUpperIndex += 1
                else:
                    break



        return outputTensor

    def getVision(self):
        """ Get vision of all cells as input for neural network """
        inputTensor = [self.viewDistance for i in range(self.rayCount)]
        for otherCell in Cell.CELL_SETS[self.getSetIndex()[0]][self.getSetIndex()[1]]:
            if otherCell.type == self.type:
                continue
            otherCellVision = self.getVisionOfCell(otherCell)
            inputTensor = [min(inputTensor[i], otherCellVision[i]) for i in range(self.rayCount)]

        return inputTensor

    def getMove(self):
        """ Feed vision input from `getVision()` into neural network """
        inputTensor = self.getVision()
        self.tensorVision = inputTensor[:]
        inputTensor.append(self.energy/self.maxEnergy)
        moveSpeed, turnSpeed = self.startingNetwork.forward(inputTensor, self.viewDistance)

        if self.type == 1:
            moveSpeed = min(self.energy, moveSpeed)
            self.energy -= moveSpeed
        self.speed = (moveSpeed * 2 - 1) * Cell.MAXIMUM_SPEED
        if self.type == 0:
            self.speed = abs(self.speed)
        self.angularVelocity = (turnSpeed * 2 - 1) * Cell.MAXIMUM_TURN_SPEED


    def update(self):
        self.getMove()
        self.move()
        self.turn()

        if self.type == 0:
            self.digestionTimer += 1
            self.energy -= 0.001
            if self.energy < 0:
                self.kill()
        else:
            self.lifeLength += 1
            self.energy += 1

    def mutate(self):
        """Randomly changes synapses in the cells neural network depending on generation"""
        self.startingNetwork.mutate(self.generationNumber)

    def draw(self, screen, drawRays = False):
        """ Draw the cell on `canvas` """
        if drawRays:
            count = 0
            for ray in self.rays:
                rayDest = (self.xyPos[0] + self.tensorVision[count]*math.cos(self.angle + ray), self.xyPos[1] + self.tensorVision[count]*math.sin(self.angle + ray))
                pyg.draw.line(screen, Cell.RAY_COLOUR, self.xyPos, rayDest, 1)
                count += 1
        pyg.draw.circle(screen, self.colour, self.xyPos, Cell.CELL_RADIUS, 0)

        #draw.line(screen, self.colour, self.xyPos, Cell.CELL_FRONT_LENGTH, 2)
        #draw outward rays

class Predator(Cell):
    MAXIMUM_DIGESTION_TIMER = 20
    MAXIMUM_ENERGY = 150
    INITIAL_ENERGY = 50
    RAY_COUNT = 15
    RAY_GAP = 0.06
    VIEW_DISTANCE = 150
    CONSUMPTION_ENERGY = 50

    def __init__(self, cellMap, inheritingNetwork = Cell.EMPTY_NETWORK, previousGenerationNumber = -1, xyPos = None):
        if xyPos == None:
            xyPos = [0, 0]
        super().__init__(cellMap, inheritingNetwork, previousGenerationNumber, xyPos)
        self.energy = Predator.INITIAL_ENERGY
        self.digestionTimer = 0
        self.type = 0
        self.colour = Cell.PREDATOR_COLOUR
        self.rays = [-Predator.RAY_GAP*(Predator.RAY_COUNT//2) + Predator.RAY_GAP*i for i in range(Predator.RAY_COUNT)]
        self.rayCount = Predator.RAY_COUNT
        self.rayGap = Predator.RAY_GAP
        self.viewDistance = Predator.VIEW_DISTANCE
        self.maxEnergy = Predator.MAXIMUM_ENERGY
        if inheritingNetwork == Cell.EMPTY_NETWORK:
            self.startingNetwork = ca.CellNet(self.rayCount)
        else:
            self.startingNetwork = inheritingNetwork
        if cuda.is_available():
            self.startingNetwork = self.startingNetwork.to("cuda:0")


    def eatPrey(self, victim):
        """ Attempt to eat `victim` """
        if self.digestionTimer >= Predator.MAXIMUM_DIGESTION_TIMER:
            victim.kill()
            self.energy += Predator.CONSUMPTION_ENERGY
            self.digestionTimer = 0
            return True
        return False
    
    def canSplit(self):
        """ Check if cell has enough energy to split """
        return self.energy >= Predator.MAXIMUM_ENERGY

    def split(self):
        if self.canSplit:
            self.energy = Predator.INITIAL_ENERGY
            self.map.dupCell(self)
            self.generationNumber += 1


class Prey(Cell):
    MAXIMUM_ENERGY = 100
    INITIAL_ENERGY = 50
    LIFESPAN = 1000
    RAY_COUNT = 14
    RAY_GAP = 0.2
    VIEW_DISTANCE = 50

    def __init__(self, cellMap, inheritingNetwork = Cell.EMPTY_NETWORK, previousGenerationNumber = -1, xyPos = None):
        if xyPos == None:
            xyPos = [0, 0]
        super().__init__(cellMap, inheritingNetwork, previousGenerationNumber, xyPos)
        self.energy = Prey.INITIAL_ENERGY
        self.type = 1
        self.lifeLength = rdm.randint(0, Prey.LIFESPAN//10)  # Marker to see when prey can split
        self.colour = Cell.PREY_COLOUR
        self.rays = [-Prey.RAY_GAP*(Prey.RAY_COUNT//2) + Prey.RAY_GAP*i for i in range(Prey.RAY_COUNT)]
        self.rayCount = Prey.RAY_COUNT
        self.rayGap = Prey.RAY_GAP
        self.viewDistance = Prey.VIEW_DISTANCE
        self.maxEnergy = Prey.MAXIMUM_ENERGY
        if inheritingNetwork == Cell.EMPTY_NETWORK:
            self.startingNetwork = ca.CellNet(self.rayCount)
        else:
            self.startingNetwork = inheritingNetwork
        if cuda.is_available():
            self.startingNetwork = self.startingNetwork.to("cuda:0")

    def canSplit(self):
        """ Check if cell has lived long enough to split """
        return self.lifeLength >= Prey.LIFESPAN

    def split(self):
        if self.canSplit:
            self.lifeLength = 0
            self.map.dupCell(self)
            self.generationNumber += 1