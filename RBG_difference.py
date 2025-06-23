import os
import rawpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

dir_path = r'C:\Users\lahir\data\blur_simulation'
files = [f for f in os.listdir(dir_path) if f.endswith('.CR2')]
wavelength = {
    'amaranth': 700,
    'amber': 599.855,
    'appleGreen':569.363,
    'arcticBlue': 482.402,
    'azure': 461.022,
    'blue': 440.020,
    'blueViolet': 422.651,
    'chartreuseGreen': 539.431,
    'green': 510.028,
    'mulberry': 412.223,
    'orchid': 405.659,
    'phthaloBlue': 443.661,
    'red': 700.000,
    'rose': 700.000,
    'sapGreen': 515.126,
    'seaGreen': 504.217,
    'tangelo': 626.205,
    'turquoise': 496.109
}

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

window_size = 10
dict={} 
for f in files:
    file_path = os.path.join(dir_path,f)
    img = read_cr2_to_rgb(file_path)
    img = np.mean(img,axis=2)
    line = img[1637,994:1787]
    line = minmax_norm(line)
    line_m = running_mean(line, window_size=window_size)

    #calc Median absolute deviation
    MAD = np.median(np.abs(line_m - np.median(line_m)))
    dict[f.split('.')[0]] = float(MAD)

pass

w,mad = [],[]
for k in dict.keys():
    if k in wavelength.keys():
        mad.append(dict[k])
        w.append(wavelength[k])

plt.scatter(w, mad, color='blue', alpha=0.5, label='Data Points')  # alpha = transparency
