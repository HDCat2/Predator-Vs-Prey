import sys
import os

currentDirectory = os.path.dirname(os.path.realpath(__file__))
parentDirectory = os.path.dirname(currentDirectory)

sys.path.append(parentDirectory)

import CellAI
import CellInformation

def test_run():
    assert(1 + 1 == 2)