from PIL import Image
import numpy as np
import os
import imageio
import cv2
import sys

sys.path.insert(0, r"C:\Users\111\Desktop\mmhealth_2\sensors")
from config import *
from postproc.tiff_to_avi import *

config = configparser.ConfigParser()
config.read( r"C:\Users\111\Desktop\mmhealth_2\sensors\configs\sensors_config.ini")

data_folder_name = r"C:\Temp\mmhealth_data\testing6"

# find all tiff files, store their paths in a list
file_list = os.listdir(data_folder_name)
#for loop, iterate through each filepath and do the conversion
for file in file_list:
    filename_ext = os.path.basename(file)
    ext = os.path.splitext(filename_ext)[1]
    if (ext == ".tiff"):
        tiff_to_avi(os.path.join(data_folder_name, file))