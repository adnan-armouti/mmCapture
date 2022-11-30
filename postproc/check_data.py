# file for postprocessing. Extracts values from data as a sanity check. 
import os
import imageio
import numpy as np
from six import string_types

import sys
sys.path.insert(0, r"C:\Users\111\Desktop\mmhealth_2\sensors")
from config import *

health_ids = ["ppg.csv"]
mic_ids = ["mic", "webcam"]

def aslist_cronly(value):
    if isinstance(value, string_types):
        value = filter(None, [x.strip() for x in value.splitlines()])
    return list(value)

def aslist(value, flatten=True):
    """ Return a list of strings, separating the input based on newlines
    and, if flatten=True (the default), also split on spaces within
    each line."""
    values = aslist_cronly(value)
    if not flatten:
        return values
    result = []
    for value in values:
        subvalues = value.split()
        result.extend(subvalues)
    return result

def get_sensor_files_list(file_path, sensor):
    sensor_files_list = []
    if (sensor == "rgb" or sensor == "thermal" or sensor == "uv" or sensor == "nir"):
        sensor_files_list.append(os.path.join(file_path, sensor + ".tiff") )
        sensor_files_list.append(os.path.join(file_path, sensor + ".txt") )
        sensor_files_list.append(os.path.join(file_path, sensor + "_local.txt") )
        if (config.getint("mmhealth", "tiff_to_avi") == 1):
            sensor_files_list.append(os.path.join(file_path, sensor + "_avi.avi") )

    elif (sensor == "polarized"):
        sensor_files_list.append(os.path.join(file_path, sensor + "_0.tiff") )
        sensor_files_list.append(os.path.join(file_path, sensor + "_45.tiff") )
        sensor_files_list.append(os.path.join(file_path, sensor + "_90.tiff") )
        sensor_files_list.append(os.path.join(file_path, sensor + "_135.tiff") )
        sensor_files_list.append(os.path.join(file_path, sensor + ".txt") )
        sensor_files_list.append(os.path.join(file_path, sensor + "_local.txt") )
        if (config.getint("mmhealth", "tiff_to_avi") == 1):
            sensor_files_list.append(os.path.join(file_path, sensor + "_0_avi.avi") )
            sensor_files_list.append(os.path.join(file_path, sensor + "_45_avi.avi") )
            sensor_files_list.append(os.path.join(file_path, sensor + "_90_avi.avi") )
            sensor_files_list.append(os.path.join(file_path, sensor + "_135_avi.avi") )
    
    elif (sensor == "rgbd"):
        sensor_files_list.append(os.path.join(file_path, sensor + "_rgb.tiff") )
        sensor_files_list.append(os.path.join(file_path, sensor + "_depth.tiff") )
        sensor_files_list.append(os.path.join(file_path, sensor + ".txt") )
        sensor_files_list.append(os.path.join(file_path, sensor + "_local.txt") )
        if (config.getint("mmhealth", "tiff_to_avi") == 1):
            sensor_files_list.append(os.path.join(file_path, sensor + "_rgb_avi.avi") )
            sensor_files_list.append(os.path.join(file_path, sensor + "_depth_avi.avi") )

    elif (sensor == "rf"):
        sensor_files_list.append(os.path.join(file_path, sensor + ".pkl") )
        sensor_files_list.append(os.path.join(file_path, sensor + ".txt") )
        sensor_files_list.append(os.path.join(file_path, sensor + "_local.txt") )
        if (config.getint("mmhealth", "read_rf_pkl") == 1):
            sensor_files_list.append(os.path.join(file_path, sensor + "_read.pkl") )

    elif (sensor == "audio"):
        sensor_files_list.append(os.path.join(file_path, sensor + ".wav") )
        sensor_files_list.append(os.path.join(file_path, sensor + ".txt") )
        sensor_files_list.append(os.path.join(file_path, sensor + "_local.txt") )

    elif (sensor == "mx800"):
        sensor_files_list.append(os.path.join(file_path, "NOM_ECG_ELEC_POTL_IIWaveExport.csv") )
        sensor_files_list.append(os.path.join(file_path, "NOM_PLETHWaveExport.csv") )
        sensor_files_list.append(os.path.join(file_path, "NOM_RESPWaveExport.csv") ) 
        sensor_files_list.append(os.path.join(file_path, "MPDataExport.csv") )
        sensor_files_list.append(os.path.join(file_path, "MPrawoutput.txt") )
        sensor_files_list.append(os.path.join(file_path, "vital_matrix.npy") )
        

    return sensor_files_list
        # if (config.getint("mmhealth", "read_rf_pkl") == 1):


def get_img_stats(video_path):
    imarray = imageio.volread(video_path)
    #sample 20 frames
    num_frames = imarray.shape[0]
    random_idxs = np.random.choice(num_frames, size = 20, replace=False)
    imarray = imarray[random_idxs]
    #load timestamps
    # print(video_path)
    path = os.path.dirname(video_path)
    filename_ext = os.path.basename(video_path)
    filename = os.path.splitext(filename_ext)[0]
    if( (filename == "rgbd_depth") or (filename == "rgbd_rgb") ):
        time_stamp_file = os.path.join(path, "rgbd_local.txt")
    elif( (filename == "polarized_0") or (filename == "polarized_45") or (filename == "polarized_90") or (filename == "polarized_135") ):
        time_stamp_file = os.path.join(path, "polarized_local.txt")
    else:
        time_stamp_file = os.path.splitext(video_path)[0] + "_local.txt"
    # print(time_stamp_file)
    time_stamps = np.genfromtxt(time_stamp_file, delimiter='\n')
    fps = num_frames / (time_stamps[-1] - time_stamps[0])
    fps_jitter = np.std(np.diff(time_stamps))

    avg_pixel = np.mean(imarray, axis=(0,1,2))
    std_pixel = np.std(imarray, axis = (0,1,2))

    # print('Average Pixel :', avg_pixel)
    # print('STD pixel :', std_pixel)
    # print('(start time , end time) : ({} , {})'.format(time_stamps[0], time_stamps[-1]))
    # print('FPS : ', fps)
    # print('FPS jitter : ', fps_jitter)

    data = [avg_pixel, std_pixel, time_stamps[0], time_stamps[-1], fps, fps_jitter]
    return data

def compare_config(sensors_dict): # compare against input config (avg_pixel_delta, std_pixel_delta [establish empirically], spatial&temporal_resolution/fps or fps_delta, NUM_FRAMES, unit8 vs uint16)
    print_counter = 0
    for sensor_type in sensors_dict:
        [avg_pixel, std_pixel, start_time, end_time, fps, fps_jitter] = sensors_dict[sensor_type]
        data_dict = {"start_time" : start_time, "end_time" : end_time, "fps" : fps, "fps_jitter" : fps_jitter, "avg_pixel" : avg_pixel, "std_pixel" : std_pixel}

        for data_key in data_dict:
            if (data_key == "fps" or data_key == "fps_jitter"):
                config_tag = "mmhealth"
                lower_bound = config.getint(config_tag, data_key) - config.getint(config_tag, data_key + "_delta")
                upper_bound = config.getint(config_tag, data_key) + config.getint(config_tag, data_key + "_delta")
            elif(data_key == "start_time"):
                config_tag = "mmhealth"
                lower_bound = 0
                upper_bound = config.getint(config_tag, data_key + "_delta")
            elif(data_key == "end_time"):
                config_tag = "mmhealth"
                data_key = "acquire_time"
                lower_bound = config.getint(config_tag, data_key) - config.getint(config_tag, data_key + "_delta")
                upper_bound = config.getint(config_tag, data_key) + config.getint(config_tag, data_key + "_delta")
                data_key = "end_time"
            else:
                config_tag = sensor_type
                lower_bound = config.getint(config_tag, data_key) - config.getint(config_tag, data_key + "_delta")
                upper_bound = config.getint(config_tag, data_key) + config.getint(config_tag, data_key + "_delta")

            data_value = data_dict[data_key]
            if (sensor_type == "rgbd" or sensor_type == "rgb" or sensor_type == "uv"):
                data_value = np.mean(data_value)
            if (np.isnan(data_value)):
                data_value = 0
            data_value = int(data_value)
            if data_value not in range(lower_bound, upper_bound):
                print("WARNING: {} {} value: {} not in expected/requested range".format(  sensor_type, data_key, data_value) )
                print("Lower Bound: {}".format( lower_bound ) )
                print("Upper Bound: {}".format(  upper_bound ) )
                print_counter += 1
        
    if (print_counter == 0):
        print("No Warnings Detected!")


def check_files_exist(sensors_dict): #TODO use rgbd data to confirm range measurement from rf
    pass

def check_data_folder(dir_path):
    sensors_str = config.get("mmhealth", "sensors_list")
    sensors_list_str = aslist(sensors_str, flatten=True)
    cameras = sensors_list_str

    for sensor in cameras:
        if (sensor != "rgbd" and  sensor != "rgb" and sensor != "polarized" and  sensor != "nir" and sensor != "thermal" and  sensor != "uv"):
            cameras.remove(sensor)

    file_list = os.listdir(dir_path)
    sensors_dict = {}

    for file in file_list:
        #check video files
        for video in cameras:
            if video in file:
                if ".tiff" in file:
                    # print(file)
                    # print("video {} stats:".format(video))
                    data = get_img_stats(os.path.join(dir_path, file))
                    sensors_dict.update({video: data})

    for sensor in sensors_list_str:
        sensor_files_list = get_sensor_files_list(dir_path, sensor)
        for file in sensor_files_list:
            if(os.path.isfile(file)):
                pass
            else:
                filename_ext = os.path.basename(file)
                print("WARNING: File {} does not exist in Path {}".format(  filename_ext, file) )
            
    print("Comparing against config parameters:")
    compare_config(sensors_dict)
    # print("Comparing sensor start and end times:")
    # compare_sensor_times(sensors_dict)

if __name__ == '__main__':


    check_data_folder(r"E:\mmhealth_data\5_13")