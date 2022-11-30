# audio sensor

import pyaudio
import wave
import sys

from config import *
from sensor import Sensor

audio = pyaudio.PyAudio()

class Audio_Sensor(Sensor):

    def __init__(self, filename : str, foldername : str = "individual_sensor_test"):
        super().__init__(filename=filename, foldername=foldername)
        
        #set parameters for audio recorder
        self.filename = filename
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = config.getint("audio", "channels")
        self.RATE = 44100#config.getint("audio", "sample_rate")
        self.CHUNK = 1024#config.getint("audio", "chunk")
        self.RECORD_SECONDS = 5
        self.WAVE_OUTPUT_FILENAME = self.filepath + ".wav"

        if (self.filename[0:2] == "mic"):
            self.input_device_index = 1
            self.sensor_type = "mic_audio"
        else: # webcam
            self.input_device_index = 2
            self.sensor_type = "webcam_audio"

        #initialize capture
        self.sensor_type = "microphone"
        self.frames = []
        self.stream = audio.open(format=self.FORMAT, channels=1, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)#, input_device_index=self.input_device_index)


    def __del__(self) -> None:
        self.release_sensor()
        print("Released {} resources.".format(self.sensor_type))
        

    def acquire(self, acquisition_time : int) -> bool:

        for i in range(0, int(self.RATE / self.CHUNK * acquisition_time)):
            self.data = self.stream.read(self.CHUNK)
            self.frames.append(self.data)
            self.record_timestamp()
            
        # stop Recording
        self.stream.stop_stream()
        self.stream.close()
        audio.terminate()

        self.save_timestamps()
        self.time_stamps = []

        self.waveFile = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        self.waveFile.setnchannels(1)
        self.waveFile.setsampwidth(audio.get_sample_size(self.FORMAT))
        self.waveFile.setframerate(self.RATE)
        self.waveFile.writeframes(b''.join(self.frames))

    def release_sensor(self) -> bool:
        self.waveFile.close()

    def print_stats(self):
        print("_____________ Audio Specifications _____________")
        print("Rate = {} 1/s".format(self.RATE))
        print("Channels = {}".format(self.CHANNELS))
        print("Recording Length: {}".format(self.RECORD_SECONDS))

#To test code, run this file.
if __name__ == '__main__':

    # Parallelize this??
    mic = Audio_Sensor(foldername= "2_1", filename="mic_audio_1") # input_device_index = 1
    # webcam = Audio_Sensor(filename="webcam_audio_1") # input_device_index = 2

    mic.acquire(acquisition_time=10)
    # webcam.acquire(acquisition_time=5)