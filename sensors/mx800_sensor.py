# mx800 sensor

import pyautogui
import time
import os
import keyboard
import sys
import shutil
import subprocess
import signal
# import msvcrt

from config import *
from sensor import Sensor

class MX800_Sensor(Sensor):

    def __init__(self, filename : str, foldername : str = "individual_sensor_test"):
        super().__init__(filename=filename, foldername=foldername)

        self.sensor_type = "mx800"

        #set parameters for video recorder
        # self.long_pause = 1
        # self.short_pause = 0.1
        self.abs_path  = r"C:\Users\111\Desktop\mmhealth_2\sensors\VSCaptureMP-master-copy\VSCaptureMP\VSCaptureMP\bin\Debug"
        self.filepath = os.path.join(self.abs_path, "VSCaptureMP.exe")
        wait_time = 1
        # self.time_delta = 3.9 # empirically determined


        self.record = subprocess.Popen([self.filepath], stdin=subprocess.PIPE) # stdout=subprocess.PIPE, stdin=subprocess.PIPE, 
        time.sleep(wait_time)
        
    def __del__(self) -> None:
        self.release_sensor()
        print("Released {} resources.".format(self.sensor_type))
        
    def acquire(self, acquisition_time : int) -> bool:
        # print("acquire start *************************")
        record_start = os.write(self.record.stdin.fileno(), chr(32).encode() )
        # print("recording start *************************")
        # time.sleep(acquisition_time + self.time_delta )
        time.sleep(acquisition_time )
        record_esc = self.record.communicate(input= chr(27).encode() )[0]
        # print("recording end *************************")

        self.release_sensor()


    def release_sensor(self) -> bool:
        #Release mx800
        pass

    def print_stats(self):
        print("_____________ mx800 Specifications _____________")
        # print("FPS = {} f/s".format(self.fps))
        # print("Resolution = {} x {}".format(self.height, self.width))
        # print("Encoding: {}".format(self.fourcc))



#To test code, run this file.
if __name__ == '__main__':
    time_delta = 3.9 # empirically determined

    mx800 = MX800_Sensor(filename="test_run")
    mx800.acquire(acquisition_time=10 + time_delta)
