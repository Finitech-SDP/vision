from PIL import Image

import sys
sys.path.append('../..')

import matplotlib.pyplot as plt

import numpy as np

from planner.app import load_coordinates, captureImage
from planner.config import COORDINATE_PATH
import cv2




def findRobot(image, coordinates,annotate):
    
    for tile, space in coordinates.items():
        spaceCropped = (image[space[0][1]:space[1][1], space[0][0]:space[1][0],:])
        
        if spaceCropped is None:
            raise Exception("coordinates wrong?")
        
        chessboardFound, _ = cv2.findChessboardCorners(spaceCropped, (9,7),None)

        if chessboardFound:
            if annotate:
                drawRect(image, space)
            return tile
    
    return None
    

def drawRect(image, space, colour=(0, 255, 0)):
    cv2.rectangle(image, (space[0][0], space[0][1]), (space[1][0], space[1][1]),
                  colour, 2)



coordinates = load_coordinates("../../" + COORDINATE_PATH)

imageCap = captureImage()

image = np.copy(imageCap)

robotTile = findRobot(image, coordinates,True)



