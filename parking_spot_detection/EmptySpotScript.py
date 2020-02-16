	
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2

def drawRect(index, image, spaces):
	cv2.rectangle(image,(spaces[index][0][0],spaces[index][0][1]),(spaces[index][1][0],spaces[index][1][1]),(0,255,0),2)


def getEmptySpaces(imagePath, showSpaces = False):
	
	image = cv2.imread(imagePath)

	spaces = np.array([
	[[ 383,217],[ 588,399]],
 	[[ 383,440],[ 588,629]],
    [[ 396,665],[ 593,837]],
 	[[ 407,883],[ 593,1016]],
	[[ 934,217],[1100,399]],
	[[ 934,427],[1100,611]],
	[[ 929,660],[1087,814]],
	[[ 921,872],[1075,988]]])

	#spaceCheck array - zero if empty 1 if theres a car in it
	#ordered by spaces on left - 0 to 3 top to bottom
	#			spaces on right -4 to 7 top to bottom

	spaceCheck = np.zeros(8)
	for i in range(8):
	    space = spaces[i]
	    spaceCropped = (image[space[0][1]:space[1][1],space[0][0]:space[1][0]])
	    gray = cv2.cvtColor(spaceCropped, cv2.COLOR_BGR2GRAY)
	    edged = cv2.Canny(gray,150,200)
	    
	    if np.sum(edged) > 100:
	        drawRect(i,image,spaces)
	        spaceCheck[i] = 1

	if showSpaces:	
		plt.imshow(image)
		plt.show()
	

	return spaceCheck


getEmptySpaces('test.jpeg', showSpaces=True)