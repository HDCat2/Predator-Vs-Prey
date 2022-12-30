import sys
import os

currentDirectory = os.path.dirname(os.path.realpath(__file__))
parentDirectory = os.path.dirname(currentDirectory)

sys.path.append(parentDirectory)

import CellAI as cai
import CellInformation as ci
import CellUtil as cu
import math
import numpy as np

def test_map_set_initialization():
    dummyMap1 = ci.Map(400, 300, 0, 0, 1, 1, False)
    cell11 = ci.Predator(dummyMap1, None, -1, [110, 100])
    cell12 = ci.Prey(dummyMap1, None, -1, [399, 280])
    viewDist = ci.Cell.VIEW_DISTANCE
    radius = ci.Cell.CELL_RADIUS
    boxCount = (dummyMap1.width//(radius*4+viewDist*2+5), dummyMap1.height//(radius*4+viewDist*2+5))

    assert(boxCount[0] == len(cell11.CELL_SETS))
    assert(boxCount[1] == len(cell11.CELL_SETS[0]))
    assert(cell11.CELL_SETS == cell12.CELL_SETS)
    assert(dummyMap1.width == (boxCount[0]*ci.Cell.BOX_SIZE[0]))
    assert(dummyMap1.height == (boxCount[1] * ci.Cell.BOX_SIZE[1]))

    dummyMap1.kill()

    dummyMap2 = ci.Map(4000, 2000, 0, 0, 1, 1, False)
    cell21 = ci.Predator(dummyMap2, None, -1, [110, 100])
    cell22 = ci.Prey(dummyMap2, None, -1, [399, 280])
    viewDist = ci.Cell.VIEW_DISTANCE
    boxCount = (dummyMap2.width//(radius*4+viewDist*2+5), dummyMap2.height//(radius*4+viewDist*2+5))

    assert(boxCount[0] == len(cell21.CELL_SETS))
    assert(boxCount[1] == len(cell21.CELL_SETS[0]))
    assert(cell21.CELL_SETS == cell22.CELL_SETS)
    assert(dummyMap2.width == (boxCount[0]*ci.Cell.BOX_SIZE[0]))
    assert(dummyMap2.height == (boxCount[1] * ci.Cell.BOX_SIZE[1]))

    dummyMap2.kill()


def test_get_set_index():
    dummyMap1 = ci.Map(400, 300, 0, 0, 1, 1, False)
    cell11 = ci.Predator(dummyMap1, None, -1, [110, 100])
    cell12 = ci.Prey(dummyMap1, None, -1, [399, 280])

    assert(cell11.getSetIndex() == (0, 0))
    assert(cell12.getSetIndex() == (len(ci.Cell.CELL_SETS)-1, len(ci.Cell.CELL_SETS[0])-1))

    dummyMap1.kill()


def test_wrap_coords():
    dummyMap1 = ci.Map(1280, 720, 0, 0, 1, 1, False)
    cell11 = ci.Cell(dummyMap1, None, -1, [10, 38])
    cell12 = ci.Cell(dummyMap1, None, -1, [1275, 704])
    cell13 = ci.Cell(dummyMap1, None, -1, [600, 400])

    assert(cell11.wrapCoords((1210, 200)) == (-70, 200))
    assert (cell11.wrapCoords((1210, 700)) == (-70, -20))
    assert (cell12.wrapCoords((5, 6)) == (1285, 726))
    assert (cell13.wrapCoords((550, 600)) == (550, 600))

    dummyMap1.kill()

def test_get_set_indices():
    dummyMap1 = ci.Map(1280, 720, 0, 0, 1, 1, False)
    cell11 = ci.Cell(dummyMap1, None, -1, [10, 38])
    cell12 = ci.Cell(dummyMap1, None, -1, [1275, 704])
    cell13 = ci.Cell(dummyMap1, None, -1, [636, 355])

    boxCount = (ci.Cell.BOX_HOR_COUNT-1, ci.Cell.BOX_VER_COUNT-1)

    assert(cell11.getSetIndices() == {(0, 0), (0, boxCount[1]), (boxCount[0], 0), boxCount})
    assert(cell12.getSetIndices() == {(0, 0), (0, boxCount[1]), (boxCount[0], 0), boxCount})
    assert(cell13.getSetIndices() == {(boxCount[0]//2, boxCount[1]//2)}
           or cell13.getSetIndices() == {(boxCount[0]//2, boxCount[1]//2), (boxCount[0]//2+1, boxCount[1]//2)}
           or cell13.getSetIndices() == {(boxCount[0]//2, boxCount[1]//2), (boxCount[0]//2, boxCount[1]//2+1)}
           or cell13.getSetIndices() == {(boxCount[0]//2, boxCount[1]//2), (boxCount[0]//2+1, boxCount[1]//2), (boxCount[0]//2, boxCount[1]//2+1), (boxCount[0]//2+1, boxCount[1]//2+1)})

    dummyMap1.kill()

def test_update_sets():
    dummyMap = ci.Map(1280, 1720, 0, 0, 1, 1, False)
    cell1 = ci.Cell(dummyMap, None, -1, [1, 1])
    cell1New = (150, 150)
    cell2 = ci.Cell(dummyMap, None, -1, [636, 355])
    cell2New = (700, 355)

    boxCount = (ci.Cell.BOX_HOR_COUNT-1, ci.Cell.BOX_VER_COUNT-1)
    indexSet11 = cell1.getSetIndices()
    indexSet12 = cell1.getSetIndices(cell1New)
    cell1.updateSets(cell1New)

    for ind in indexSet11:
        if ind not in indexSet12:
            assert(not cell1 in cell1.CELL_SETS[ind[0]][ind[1]])


    for ind in indexSet12:
        assert(not cell1 not in cell1.CELL_SETS[ind[0]][ind[1]])

    indexSet21 = cell2.getSetIndices()
    indexSet22 = cell2.getSetIndices(cell2New)
    cell2.updateSets(cell2New)

    for ind in indexSet21:
        if ind not in indexSet22:
            assert(not cell2 in cell2.CELL_SETS[ind[0]][ind[1]])

    for ind in indexSet22:
        assert(not cell2 not in cell2.CELL_SETS[ind[0]][ind[1]])

    dummyMap.kill()

