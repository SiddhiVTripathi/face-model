import numpy as np
import cv2
def rgb2ir(image):
    #0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue.
    r,g,b = cv2.split(image)
    # return (r*.6)+(g*.3)+(b*.1)
    return cv2.addWeighted(cv2.addWeighted(r, 2.9/3, g, 0.1/3, -1),0.99,b,0.01,-1)
