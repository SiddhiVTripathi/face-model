import numpy as np
from rgb2ir import rgb2ir
# from PIL import Image
import cv2
img = cv2.imread('photo.jpg')
im = rgb2ir(img)
cv2.imshow('image', im)
cv2.imshow('image_bw',cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cv2.waitKey(0)
cv2.destroyAllWindows()