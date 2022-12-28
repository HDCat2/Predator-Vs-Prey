import numpy as np

def minDistanceOfRayFromPoint(rayOrigin, rayDirection, pointCoords):
    """ Calculate the minimum distance a given ray is from a given point """
    var1 = -1 * ( (rayOrigin[0] - pointCoords[0]) * rayDirection[0] + (rayOrigin[1] - pointCoords[1]) * rayDirection[1] )
    var2 = var1 / (rayDirection[0]**2 + rayDirection[1]**2)
    var3 = ((rayOrigin[0] + var2 * rayDirection[0] - pointCoords[0])**2 + (rayOrigin[1] + var2 * rayDirection[1] - pointCoords[1])**2)**0.5
    return var3