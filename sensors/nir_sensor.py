import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio
import PyCapture2

from sensor import Sensor
from config import *

class NIR_Sensor(Sensor):

    def __init__(self, filename : str, foldername : str = "individual_sensor_test"):
        super().__init__(filename=filename, foldername=foldername)
        
        #sensor info
        self.sensor_type = "nir_camera"
        self.fps     = config.getint("mmhealth", "fps")
        self.width   = config.getint("nir", "width")
        self.height  = config.getint("nir", "height")
        self.channels = config.getint("nir", "channels")
        self.compression = config.getint("nir", "compression")
        self.calibrate_mode = config.getint("mmhealth", "calibration_mode") 
        self.calibrate_format = ".png"

        # Ensure sufficient cameras are found
        bus = PyCapture2.BusManager()
        num_cams = bus.getNumOfCameras()
        # print('Number of cameras detected: %d' % num_cams)
        if not num_cams:
            print('Insufficient number of cameras. Exiting...')
            exit()
        # Select camera on 0th index
        self.cam_nir = PyCapture2.Camera()
        self.cam_nir.connect(bus.getCameraFromIndex(0))


        # set height and width
        if (self.height != 2048 and self.width != 2048):
            fmt7imgSet = PyCapture2.Format7ImageSettings(1, 0, 0, width = self.width, height = self.height)
            fmt7pktInf, isValid = self.cam_nir.validateFormat7Settings(fmt7imgSet)
            self.cam_nir.setFormat7ConfigurationPacket(fmt7pktInf.recommendedBytesPerPacket, fmt7imgSet)
        else:
            fmt7imgSet = PyCapture2.Format7ImageSettings(0, 0, 0, width = self.width, height = self.height)
            fmt7pktInf, isValid = self.cam_nir.validateFormat7Settings(fmt7imgSet)
            self.cam_nir.setFormat7ConfigurationPacket(fmt7pktInf.recommendedBytesPerPacket, fmt7imgSet)

        # set fps
        self.cam_nir. setProperty(type = PyCapture2.PROPERTY_TYPE.FRAME_RATE, autoManualMode = False)
        self.cam_nir.setProperty(type = PyCapture2.PROPERTY_TYPE.FRAME_RATE, onOff = True)
        self.cam_nir.setProperty(type = PyCapture2.PROPERTY_TYPE.FRAME_RATE, absValue = self.fps)
        # fRateProp = self.cam_nir.getProperty(PyCapture2.PROPERTY_TYPE.FRAME_RATE)
        # self.fps = int(fRateProp.absValue)
        self.format = ".tiff"

        # print('Starting capture...')
        self.cam_nir.startCapture()

    def __del__(self) -> None:
        self.release_sensor()
        print("Released {} resources.".format(self.sensor_type))

    def acquire(self, acquisition_time : int) -> bool:
        if (self.calibrate_mode == 1):
            run = True
            while( run == True):
                try:
                    image = self.cam_nir.retrieveBuffer()
                    im_arr = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()) )
                    frame = cv2.resize(im_arr, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
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
                    # #     run = False
                    # #     break
                except PyCapture2.Fc2error as fc2Err:
                    print('Error retrieving buffer : %s' % fc2Err)
                    continue
                

        else:
            NUM_FRAMES = self.fps*acquisition_time  # number of images to capture
            frames = np.empty((NUM_FRAMES, self.height, self.width), np.dtype('uint8'))
            # video = PyCapture2.FlyCapture2Video()
            # video.AVIOpen((self.filepath + self.format).encode('utf-8'), self.fps)
            for i in range(NUM_FRAMES):
                try:
                    image = self.cam_nir.retrieveBuffer()
                    self.record_timestamp()
                    im_arr = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols()) )
                except PyCapture2.Fc2error as fc2Err:
                    print('Error retrieving buffer : %s' % fc2Err)
                    continue
                frames[i] = im_arr

            imageio.mimwrite(self.filepath + self.format, frames, bigtiff=True)

            self.save_timestamps()
            self.time_stamps = []
            
    def release_sensor(self) -> bool:
        # Deinitialize camera
        self.cam_nir.stopCapture()
        self.cam_nir.disconnect()

    def print_stats(self):
        print("_____________ NIR Camera Specifications _____________")
        cam_info = self.cam_nir.getCameraInfo()
        print('Serial number - ', cam_info.serialNumber)
        print('Camera model - ', cam_info.modelName)
        print('Camera vendor - ', cam_info.vendorName)
        print('Sensor - ', cam_info.sensorInfo)
        print('Resolution - ', cam_info.sensorResolution)
        print('FPS - ', self.fps)
        print('Channels - ', self.channels)
        print('Compression - ', self.compression)
        print('Firmware version - ', cam_info.firmwareVersion)
        print('Firmware build time - ', cam_info.firmwareBuildTime)
        print()

# #To test code, run this file.
if __name__ == '__main__':

    nir_cam = NIR_Sensor(filename="nir_output_2")
    nir_cam.acquire(acquisition_time=5)