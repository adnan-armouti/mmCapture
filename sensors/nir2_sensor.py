import sys
import os
import PySpin
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio
import multiprocessing as mp
import time
from datetime import datetime

from sensor import Sensor
from config import *

class NIR2_Sensor(Sensor):

    def __init__(self, filename : str, foldername : str = "individual_sensor_test"):
        super().__init__(filename=filename, foldername=foldername)

        self.sensor_type = "nir2_camera"
        self.fps     = config.getint("mmhealth", "fps")
        self.width   = config.getint("nir2", "width") 
        self.height  = config.getint("nir2", "height") 
        self.channels = config.getint("nir2", "channels") 
        self.compression = config.getint("nir2", "compression")
        self.calibrate_mode = config.getint("mmhealth", "calibration_mode") 
        self.calibrate_format = ".png"
        self.format = ".tiff"

        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        self.num_cameras = self.cam_list.GetSize()

        if self.num_cameras == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            print('Not enough cameras!')

        for i, self.cam in enumerate(self.cam_list):
            nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
                self.device_serial_number = node_device_serial_number.GetValue()

            if (self.device_serial_number == "19224369"): # TODO
                self.cam_nir2 = self.cam
                del self.cam
            elif (self.device_serial_number == "21290846"):
                del self.cam

        self.cam_nir2.Init()
        self.nodemap = self.cam_nir2.GetNodeMap()
        self.nodemap_tldevice = self.cam_nir2.GetTLDeviceNodeMap()

        # set fps
        node_acquisition_frame_rate_control_enable = PySpin.CBooleanPtr(self.nodemap.GetNode("AcquisitionFrameRateEnable"))
        if not PySpin.IsAvailable(node_acquisition_frame_rate_control_enable) or not PySpin.IsWritable(node_acquisition_frame_rate_control_enable):
            print("Unable to turn on Acquisition Frame Rate Control Enable (bool retrieval). Aborting...")
            return False
        node_acquisition_frame_rate_control_enable.SetValue(True)
        if self.cam_nir2.AcquisitionFrameRate.GetAccessMode() != PySpin.RW:
            print ("Unable to set Frame Rate. Aborting...")
            return False
        self.cam_nir2.AcquisitionFrameRate.SetValue(self.fps) # frame rate + 1

        node_acquisition_mode = PySpin.CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
            return False
        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(node_acquisition_mode_continuous):
            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
            return False

        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

        # self.fps = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
        # self.fps = self.fps.GetValue()
        # self.fps = int(self.fps)

        # set width
        if self.cam_nir2.Width.GetAccessMode() == PySpin.RW and self.cam_nir2.Width.GetInc() != 0 and self.cam_nir2.Width.GetMax != 0:
            self.cam_nir2.Width.SetValue(self.width)
            # print("Width set to %i..." % self.cam_nir2.Width.GetValue() )
        else:
            print("Width not available...")
            result = False

        # set height
        if self.cam_nir2.Height.GetAccessMode() == PySpin.RW and self.cam_nir2.Height.GetInc() != 0 and self.cam_nir2.Height.GetMax != 0:
            self.cam_nir2.Height.SetValue(self.height)
            # print("Height set to %i..." % self.cam_nir2.Height.GetValue() )
        else:
            print("Height not available...")
            result = False

    def __del__(self) -> None:
        self.release_sensor()
        print("Released {} resources.".format(self.sensor_type))

    def acquire(self, acquisition_time : int) -> bool:
        self.cam_nir2.BeginAcquisition()
        if (self.calibrate_mode == 1):
            run = True
            while( run == True):
                # self.cam_nir2.BeginAcquisition()
                image_result = self.cam_nir2.GetNextImage(1000)
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d...' % image_result.GetImageStatus())
                else:
                    image_data = (
                        image_result.GetData()
                        .astype(np.uint8)
                        .reshape((image_result.GetHeight(), -1))
                    )
                    image_result.Release()

                    key = cv2.waitKey(1)
                    if key == ord('s'):
                        start_num = 1
                        while(os.path.exists(self.filepath + "_0_" + str(start_num) + self.calibrate_format)):
                            start_num += 1
                        imageio.imwrite(self.filepath + str(start_num) + self.calibrate_format, image_data)
                    elif key == ord('q'):
                        run = False
                        cv2.destroyAllWindows()
                        break

                    image_result.Release()
            self.cam_nir2.EndAcquisition()

        else:
            NUM_FRAMES = self.fps*acquisition_time  # number of images to capture
            # frames = np.empty((NUM_FRAMES, int(self.height/2), int(self.width/2) ), np.dtype('uint8'))
            frames = np.empty((NUM_FRAMES, self.height, self.width ), np.dtype('uint8'))
            # self.cam_nir2.BeginAcquisition()
            for i in range(NUM_FRAMES):
                image_result = self.cam_nir2.GetNextImage(1000)
                self.record_timestamp()
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d...' % image_result.GetImageStatus())
                else:
                    # image_data = image_result.GetNDArray()
                    image_data = (
                        image_result.GetData()
                        .astype(np.uint8)
                        .reshape((image_result.GetHeight(), -1))
                    )
                    image_result.Release()
                    frames[i] = image_data

            self.cam_nir2.EndAcquisition()
            # imageio.mimwrite(self.filepath + self.format, frames, bigtiff=True)
            imageio.mimwrite(self.filepath + "_0" + self.format, frames, bigtiff=True)
            self.save_timestamps()
            self.time_stamps = []

    def acquire_save_multiprocess(self, acquisition_time : int, save_time : int) -> bool:
        # Instantiate a pipe for getting data for saving
        parent_conn, child_conn = mp.Pipe()
        SAVE_FRAMES = self.fps*save_time  # number of images to capture
        # Start the acquisition
        TOTAL_NUM_FRAMES = int(self.fps*acquisition_time)  # number of images to capture
        
        # Main part for acquire
        ded_save_1 = mp.Process(target=nir_dedicated_save, \
                        args=(SAVE_FRAMES, child_conn, self.height, self.width, self.filepath, self.format))
        ded_save_1.start()
        for i in range(TOTAL_NUM_FRAMES):
                image_result = self.cam_nir2.GetNextImage(1000)
                process_time = time.perf_counter()
                global_time = str(datetime.now())
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d...' % image_result.GetImageStatus())
                else:
                    # image_data = image_result.GetNDArray()
                    image_data = (
                        image_result.GetData()
                        .astype(np.uint8)
                        .reshape((image_result.GetHeight(), -1))
                    )
                    image_result.Release()
                parent_conn.send(image_data)
                parent_conn.send(process_time)
                parent_conn.send(global_time)
        parent_conn.close()
        
        # Hold if any process is alive
        ded_save_1.join()
            
    def release_sensor(self) -> bool:
        self.cam_nir2.DeInit()
        del self.cam_nir2
        self.cam_list.Clear()
        self.system.ReleaseInstance()

    def print_stats(self):
        print("_____________ NIR2 Camera Specifications _____________")
        print("FPS = {} f/s".format(self.fps))
        print("Resolution = {} x {}".format(self.height, self.width))
        print("Comression - ", self.compression)

def nir_dedicated_save(SAVE_FRAMES, child_conn, height, width, filepath, format):
    count = 0
    while True:
        time_stamps = []
        local_time_stamps = []
        count += 1
        frames = np.empty((SAVE_FRAMES, height, width), np.dtype('uint8'))
        # Save the frames in the specified files
        for i in range(SAVE_FRAMES):
            try:
                frames[i] = child_conn.recv().reshape((height, width))
                time_stamps.append(child_conn.recv())
                local_time_stamps.append(child_conn.recv())
            except EOFError:
                return
        imageio.mimwrite(filepath + f'_{count}' + format, frames, bigtiff=True)
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
    polarized_cam = NIR2_Sensor(filename="nir2_")
    polarized_cam.acquire(acquisition_time=5)