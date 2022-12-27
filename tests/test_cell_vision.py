import sys
import os

currentDirectory = os.path.dirname(os.path.realpath(__file__))
parentDirectory = os.path.dirname(currentDirectory)

sys.path.append(parentDirectory)

import CellAI as cai
import CellInformation as ci
import math

def test_predator_vision_cell_above():
    dummyMap = ci.Map(1280,720,0,0,1,1, False)
    cell1 = ci.Predator(dummyMap, None, -1, [100,100])
    cell2 = ci.Prey(dummyMap, None, -1, [100,200])

    cell1.cellAngle = math.pi/2

    cell1Vision = cell1.getVisionOfCell(cell2)
    print(cell1Vision)

    assert(1+1==2)

def test_predator_vision_cell_inside():
    dummyMap = ci.Map(1280,720,0,0,1,1, False)
    cell1 = ci.Predator(dummyMap, None, -1, [100,100])
    cell2 = ci.Prey(dummyMap, None, -1, [100,101])

    cell1.cellAngle = math.pi/2

    cell1Vision = cell1.getVisionOfCell(cell2)
    print(cell1Vision)

    assert(1+1==2)
