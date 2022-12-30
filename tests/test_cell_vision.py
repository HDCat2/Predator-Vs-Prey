""" A collection of unit tests for cell vision calculation and its associated methods """

import sys
import os

currentDirectory = os.path.dirname(os.path.realpath(__file__))
parentDirectory = os.path.dirname(currentDirectory)

sys.path.append(parentDirectory)

import CellInformation as ci
import CellUtil as cu
import math
import numpy as np

def test_find_min_distance_intersect_cardinal_directions():
    assert(cu.minDistanceOfRayFromPoint([100,100], [0,1], [100,200]) == 0)
    assert(cu.minDistanceOfRayFromPoint([100,100], [0,-1], [100,0]) == 0)
    assert(cu.minDistanceOfRayFromPoint([100,100], [1,0], [200,100]) == 0)
    assert(cu.minDistanceOfRayFromPoint([100,100], [-1,0], [0,100]) == 0)

def test_find_min_distance_intersect_diagonal_directions():
    assert(cu.minDistanceOfRayFromPoint([100,100], [1,1], [200,200]) == 0)
    assert(cu.minDistanceOfRayFromPoint([100,100], [-1,1], [0,200]) == 0)
    assert(cu.minDistanceOfRayFromPoint([100,100], [1,-1], [200,0]) == 0)
    assert(cu.minDistanceOfRayFromPoint([100,100], [-1,-1], [0,0]) == 0)

def test_find_min_distance_no_intersect():
    assert(cu.minDistanceOfRayFromPoint([100,100], [1,0], [200,150]) == 50)

    dist = cu.minDistanceOfRayFromPoint([100,100], [math.cos(math.pi/2 - 0.12), math.sin(math.pi/2 - 0.12)], [100,200])
    assert(abs(dist - 11.97122) < 0.00001)

    dist = cu.minDistanceOfRayFromPoint([100,100], [math.cos(math.pi/2 + 0.12), math.sin(math.pi/2 + 0.12)], [100,200])
    assert(abs(dist - 11.97122) < 0.00001)


def test_predator_vision_cell_above():
    dummyMap = ci.Map(1280,720,0,0,1,1, False)
    cell1 = ci.Predator(dummyMap, None, -1, [100,100])
    cell2 = ci.Prey(dummyMap, None, -1, [100,200])

    cell1.angle = math.pi/2

    cell1Vision = cell1.getVisionOfCell(cell2)
    almostCorrectAns = [0, 0, 0, 0, 0, 0, 0.08182644255778426, 0.1, 0.08182644255778426, 0, 0, 0, 0, 0, 0]
    assert(np.linalg.norm(np.subtract(cell1Vision, almostCorrectAns)) < 0.00001)


def test_predator_vision_cell_inside():
    dummyMap = ci.Map(1280,720,0,0,1,1, False)
    cell1 = ci.Predator(dummyMap, None, -1, [100,100])
    cell2 = ci.Prey(dummyMap, None, -1, [100,101])

    cell1.cellAngle = math.pi/2

    cell1Vision = cell1.getVisionOfCell(cell2)
    assert(cell1Vision == [1 for i in range(15)])
