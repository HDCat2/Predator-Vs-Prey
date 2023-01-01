""" A collection of unit tests for cell vision calculation and its associated methods """

import sys
import os

currentDirectory = os.path.dirname(os.path.realpath(__file__))
parentDirectory = os.path.dirname(currentDirectory)

sys.path.append(parentDirectory)

import CellInformation as ci
import numpy as np

def test_collision_same_position():
    dummyMap = ci.Map(1280, 720, 0, 0, 100, 100, False)
    cell1 = ci.Predator(dummyMap, 0, 0, [100,100])
    cell2 = ci.Predator(dummyMap, 0, 0, [100,100])

    cell1.isColliding(cell2)

    assert(cell1.collisionModifier == [0,100])
    assert(cell2.collisionModifier == [0,-100])

def test_collision_nonintersect():
    dummyMap = ci.Map(1280, 720, 0, 0, 100, 100, False)
    cell1 = ci.Predator(dummyMap, 0, 0, [100,100])
    cell2 = ci.Predator(dummyMap, 0, 0, [200,200])

    cell1.isColliding(cell2)

    assert(cell1.collisionModifier == [0,0])
    assert(cell2.collisionModifier == [0,0])

def test_collision_partial():
    dummyMap = ci.Map(1280, 720, 0, 0, 100, 100, False)
    cell1 = ci.Predator(dummyMap, 0, 0, [100,100])
    cell2 = ci.Predator(dummyMap, 0, 0, [100,110])

    cell1.isColliding(cell2)

    assert(cell1.collisionModifier == [0,-50])
    assert(cell2.collisionModifier == [0,50])

    cell3 = ci.Predator(dummyMap, 0, 0, [100,100])
    cell4 = ci.Predator(dummyMap, 0, 0, [105,105])

    cell3.isColliding(cell4)

    assert(np.linalg.norm(np.subtract(cell3.collisionModifier, [-45.710678118654755, -45.710678118654755])) < 0.0001)
    assert(np.linalg.norm(np.subtract(cell4.collisionModifier, [45.710678118654755, 45.710678118654755])) < 0.0001)