from PIL import Image
import numpy as np
import os
import imageio
import cv2
import sys

sys.path.insert(0, r"C:\Users\111\Desktop\osa\sensors")
from config import *

# input_filepath = str(sys.argv[1])
def tiff_to_avi(input_filepath):
    path = os.path.dirname(input_filepath)
    filename_ext = os.path.basename(input_filepath)
    filename = os.path.splitext(filename_ext)[0]
    # ext = os.path.splitext(filename_ext)[1]
    # print(ext)

    sensor_type = filename[:2]
    if(sensor_type == "ni"):
        sensor_type = "nir"
    elif(sensor_type == "th"):
        sensor_type = "thermal"
    else: 
    # elif (sensor_type == "uv"):
        pass
    # elif(sensor_type == "rf"):
    #     pass
    # elif(sensor_type == "au"):
    #     sensor_type = "audio"

    # sensor_type = "nir"
    width = config.getint(sensor_type, "width") 
    height = config.getint(sensor_type, "height") 
    fps = config.getint("mmhealth", "fps")

    imarray = imageio.volread(input_filepath) 
    imarray = imarray.astype("uint8")
    # imarray = imarray.astype("uint8") #uint8
    NUM_FRAMES = imarray.shape[0]

    if(NUM_FRAMES == 1):
        frame = imarray[0]
        output_filepath = os.path.join(path, filename + ".jpeg")
        if (sensor_type == "nir" or sensor_type == "thermal"):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(output_filepath, frame.astype(np.uint8))
        else:
            cv2.imwrite(output_filepath, frame.astype(np.uint8))
    else:
        output_filepath = os.path.join(path, filename + "_avi.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
        if (sensor_type == "nir" or sensor_type == "thermal"):
            for i in range (NUM_FRAMES):
                frame = imarray[i]
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                video.write(frame)
            video.release()   
        else:
            for i in range (NUM_FRAMES):
                frame = imarray[i]
                video.write(frame)
            video.release()        
    
    print("Tiff to Avi Conversion: Sensor {} done! Shape: {}".format(sensor_type, imarray.shape) )
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_filepath = r"E:\osa_data\5_9\thermal_1.tiff"
    tiff_to_avi(input_filepath)
    
# im = Image.open(filename)
# im.show()