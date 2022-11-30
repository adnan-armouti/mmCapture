# thermal sensor

import imageio
import numpy as np
import sys
import os
import cv2

from config import *
from sensor import Sensor

class Thermal_Sensor(Sensor):

    def __init__(self, filename : str, foldername : str = "individual_sensor_test"):
        super().__init__(filename=filename, foldername=foldername)

        self.sensor_type = "thermal_camera"
        
        #initialize capture
        self.format = ".tiff"
        self.fps     = config.getint("mmhealth", "fps")
        self.width   = config.getint("thermal", "width") #not setting
        self.height  = config.getint("thermal", "height") #not setting
        self.channels = config.getint("thermal", "channels") #not setting
        self.compression = config.getint("thermal", "compression")
        self.calibrate_mode = config.getint("mmhealth", "calibration_mode") 
        self.calibrate_format = ".png"
        self.counter = 0

        kargs = { 'fps': self.fps, 'ffmpeg_params': ['-s',str(self.width) + 'x' + str(self.height)] }
        self.reader = imageio.get_reader('<video1>', format = "FFMPEG", dtype = "uint16", fps = self.fps)

    def __del__(self) -> None:
        self.release_sensor()
        print("Released {} resources.".format(self.sensor_type))
        # print(self.filepath)
        
    def acquire(self, acquisition_time : int) -> bool:
        if (self.calibrate_mode == 1):
            for im in self.reader:
                upsampled_frame = im[:,:,0]
                downsampled_frame = upsampled_frame[::2,::2]
                im_arr = downsampled_frame.astype(np.uint8)
                im_arr = cv2.cvtColor(im_arr, cv2.COLOR_GRAY2RGB)
                frame = cv2.resize(im_arr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_AREA)
                cv2.imshow('Input', frame)

                key = cv2.waitKey(1)
                if key == ord('s'):
                    start_num = 1
                    while(os.path.exists(self.filepath + "_" + str(start_num) + self.calibrate_format)):
                        start_num += 1
                    imageio.imwrite(self.filepath + "_" + str(start_num) + self.calibrate_format, im_arr)
                elif key == ord('q'):
                    run = False
                    cv2.destroyAllWindows()
                    break

                # c = cv2.waitKey(1)
                # if c == 27:
                #     break
                # elif cv2.waitKey(1) & 0xFF == ord('s'):
                #     start_num = 1
                #     while(os.path.exists(self.calibrate_filepath + str(start_num) + ".png")):
                #         start_num += 1
                #     imageio.imwrite(self.calibrate_filepath + str(start_num) + ".png", im_arr)
                # # elif cv2.waitKey(1) & 0xFF == ord('q'):
                # #     break
        else:
            NUM_FRAMES = self.fps*acquisition_time  # number of images to capture
            frames = np.empty((NUM_FRAMES, self.height, self.width), np.dtype('uint16'))

            for im in self.reader:
                if (self.counter < NUM_FRAMES):
                    if ((self.counter != 0)):
                        upsampled_frame = im[:,:,0]
                        downsampled_frame = upsampled_frame[::2,::2]
                        if ( np.max(downsampled_frame  - frames[self.counter-1]) != 0 ):
                            frames[self.counter] = downsampled_frame # Reads 3 channels, but each channel is identical (same pixel info)
                            self.record_timestamp()
                            self.counter += 1
                    else:
                        upsampled_frame = im[:,:,0]
                        frames[self.counter] = upsampled_frame[::2,::2] # Reads 3 channels, but each channel is identical (same pixel info)
                        self.record_timestamp()
                        self.counter += 1
                else:
                    break

            imageio.mimwrite(self.filepath + self.format, frames, bigtiff=True)

            self.save_timestamps()
            self.time_stamps = []

        
    def release_sensor(self) -> bool:
        #Release camera
        pass

    def print_stats(self):
        print("_____________ Thermal Camera Specifications _____________")
        print("FPS Requested = {} f/s".format(self.fps))
        print("FPS Recorded = {} f/s".format(int(self.reader.get_meta_data()['fps'])))
        print("Resolution = {} x {}".format(self.width, self.height))

# #To test code, run this file.
if __name__ == '__main__':

    thermal_cam = Thermal_Sensor(filename="thermal_1")
    thermal_cam.acquire(acquisition_time=0.1)
    thermal_cam.print_stats()

    