# thermal sensor
import multiprocessing as mp
import imageio
import numpy as np
import sys
import os
import cv2
from datetime import datetime
import time

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
        self.reader = imageio.get_reader('<video0>', format = "FFMPEG", dtype = "uint16", fps = self.fps)

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

    def buffer(self):
        NUM_FRAMES = 1  # number of images to capture
        frames = np.empty((NUM_FRAMES, self.height, self.width), np.dtype('uint16'))
        for im in self.reader:
            frames[self.counter] = im[:,:,0]
            self.record_timestamp()
            break
        self.time_stamps = []

    def acquire_save_multiprocess(self, acquisition_time : int, save_time : int) -> bool:
        NUM_FRAMES = self.fps*acquisition_time  # number of images to capture
        parent_conn, child_conn = mp.Pipe()
        SAVE_FRAMES = self.fps*save_time  # number of images to capture
        ded_save_1 = mp.Process(target=thermal_dedicated_save, \
                        args=(SAVE_FRAMES, child_conn, self.height, self.width, self.filepath, self.format))
        ded_save_1.start()
        prev_frame = np.zeros((self.height, self.width), np.dtype('uint16'))
        for im in self.reader:
            process_time = time.perf_counter()
            global_time = str(datetime.now())
            if (self.counter < NUM_FRAMES):
                # upsampled_frame = im[:,:,0]
                # downsampled_frame = upsampled_frame[::2,::2]
                downsampled_frame = im[:,:,0]
                if ( np.max(downsampled_frame - prev_frame) != 0 ):
                    parent_conn.send(downsampled_frame) # Reads 3 channels, but each channel is identical (same pixel info)
                    parent_conn.send(process_time)
                    parent_conn.send(global_time)
                    self.counter += 1
                    prev_frame = downsampled_frame
            else:
                break
        parent_conn.close()
        ded_save_1.join()

    def release_sensor(self) -> bool:
        #Release camera
        pass

    def print_stats(self):
        print("_____________ Thermal Camera Specifications _____________")
        print("FPS Requested = {} f/s".format(self.fps))
        print("FPS Recorded = {} f/s".format(int(self.reader.get_meta_data()['fps'])))
        print("Resolution = {} x {}".format(self.width, self.height))

def thermal_dedicated_save(SAVE_FRAMES, child_conn, height, width, filepath, format):
    count = 0
    while True:
        time_stamps = []
        local_time_stamps = []
        count += 1
        fp = np.memmap(filepath + f'_{count}.dat', dtype='uint16', mode='w+', shape=(SAVE_FRAMES, height, width))
        # Save the frames in the specified files
        for i in range(SAVE_FRAMES):
            try:
                frame = child_conn.recv().reshape((height, width))
                fp[i] = frame
                fp.flush()
                time_stamps.append(child_conn.recv())
                local_time_stamps.append(child_conn.recv())
            except EOFError:
                return
        # Save the time stamps in the specified files
        try:
            with open(filepath + f"_{count}.txt", "w") as output:
                output.write('\n'.join([str(stamp) for stamp in time_stamps]))
            with open(filepath + f"_{count}_local.txt", "w") as output:
                output.write('\n'.join([stamp for stamp in local_time_stamps]))
        except:
            print("failed time stamp saving")
            return

# #To test code, run this file.
if __name__ == '__main__':

    thermal_cam = Thermal_Sensor(filename="thermal_1")
    thermal_cam.acquire(acquisition_time=0.1)
    thermal_cam.print_stats()

    