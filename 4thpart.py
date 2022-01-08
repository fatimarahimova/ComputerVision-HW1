import cv2
import numpy as np

image = cv2.imread("HW1_material/album.png")
degree = np.radians(60)
cols,rows,a = image.shape

centerx=cols/2
centery=rows/2

def rotation_matrix(angle, x, y):
    rotation_matrix=[[np.cos(angle), -np.sin(angle), (x-(x*np.cos(angle))+(y*np.sin(angle)))], [np.sin(angle), np.cos(angle), y - (y*np.cos(angle)) - (x*np.sin(angle))]]
    return rotation_matrix

matrix1= rotation_matrix(degree, centerx, centery)
img1 = cv2.warpAffine(image, np.array(matrix1), (cols,rows))
cv2.imwrite('centerrotated.png',img1)

matrix2=rotation_matrix(degree,0,0)
img2=cv2.warpAffine(image, np.array(matrix2),(cols,rows))
cv2.imwrite('leftcornerrotated.png', img2)

