import os
import rawpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

red_path = r'C:\Users\lahir\data\blur_simulation\RED.CR2'
green_path = r'C:\Users\lahir\data\blur_simulation\GREEN.CR2'
blue_path = r'C:\Users\lahir\data\blur_simulation\BLUE.CR2'
white_path = r'C:\Users\lahir\data\blur_simulation\WHITE.CR2'


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.show(block=True)

def read_cr2_to_rgb(cr2_path):
    # Read .CR2 file
    with rawpy.imread(cr2_path) as raw:
        # Post-process to RGB (adjust parameters as needed)
        rgb = raw.postprocess(use_camera_wb=True)    
    return rgb  # Returns a NumPy array (HxWx3)

def minmax_norm(ar):
    min_val = ar.min()
    max_val = ar.max()
    ar_norm = (ar-min_val)/(max_val-min_val)
    return ar_norm

def running_mean(arr, window_size):
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')



red_img = read_cr2_to_rgb(red_path)
red_img = np.mean(red_img,axis=2)
green_img = read_cr2_to_rgb(green_path)
green_img = np.mean(green_img,axis=2)
blue_img = read_cr2_to_rgb(blue_path)
blue_img = np.mean(blue_img,axis=2)
white_img = read_cr2_to_rgb(white_path)
white_img = np.mean(white_img,axis=2)

window_size = 1

red_line = red_img[1080,850:1720]
red_line = minmax_norm(red_line)
red_line = running_mean(red_line, window_size=window_size)

green_line = green_img[1080,850:1720]
green_line = minmax_norm(green_line)
red_line = running_mean(red_line, window_size=window_size)

blue_line = blue_img[1080,850:1720]
blue_line = minmax_norm(blue_line)
red_line = running_mean(red_line, window_size=window_size)

white_line = white_img[1080,850:1720]
white_line = minmax_norm(white_line)
red_line = running_mean(red_line, window_size=window_size)

#convert to grayscale


show_img(red_img)

pass

plt.plot(red_line, color='red',   label='Red Line')
plt.plot(green_line, color='green',   label='Green Line')
plt.plot(blue_line, color='blue',   label='Blue Line')
plt.plot(white_line, color='black',   label='White Line')
plt.show(block=True)
