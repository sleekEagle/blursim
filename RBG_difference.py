import os
import rawpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

red_path = r'C:\Users\lahir\data\blur_simulation\red.CR2'
green_path = r'C:\Users\lahir\data\blur_simulation\green.CR2'
blue_path = r'C:\Users\lahir\data\blur_simulation\blue.CR2'


def show_img(img):
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show(block=True)

def read_cr2_to_rgb(cr2_path):
    # Read .CR2 file
    with rawpy.imread(cr2_path) as raw:
        # Post-process to RGB (adjust parameters as needed)
        rgb = raw.postprocess(use_camera_wb=True)    
    return rgb  # Returns a NumPy array (HxWx3)





red_img = cv2.cvtColor(read_cr2_to_rgb(red_path),cv2.COLOR_BGR2GRAY)
green_img = cv2.cvtColor(read_cr2_to_rgb(green_path),cv2.COLOR_BGR2GRAY)
blue_img = cv2.cvtColor(read_cr2_to_rgb(blue_path),cv2.COLOR_BGR2GRAY)

red_line = red_img[1190,2044:2170]
green_line = green_img[1190,2044:2170]
blue_line = blue_img[1190,2044:2170]

show_img(red_img)

pass

plt.plot(red_line)
plt.plot(green_line)
plt.plot(blue_line)
plt.show(block=True)
