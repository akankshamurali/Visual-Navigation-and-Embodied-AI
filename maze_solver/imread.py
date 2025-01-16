import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('C:\\Desktop\\NYU\\3rd_sem\\Perception\\Midterm_project\\Maze_out.jpg', cv2.IMREAD_GRAYSCALE)
print(img.shape)
plt.imshow(img)
plt.show()
print(img[0,0])