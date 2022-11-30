import sensors.rf_UDP.organizer_copy as org
from sensors.config import *
import pickle
import os
from natsort import natsorted, ns
import imageio
import numpy as np

# def read_pickle_rf(folder_name):
#     file_list = os.listdir(folder_name)
#     for file in file_list:
#         filename_ext = os.path.basename(file)
#         filename, ext = os.path.splitext(filename_ext)
#         if (ext == ".pkl"):
#             f = open(os.path.join(folder_name , filename_ext),'rb')
#             s = pickle.load(f)
#             o = org.Organizer(s, 1, 4, 3, 512)
#             frames = o.organize()
#             print("Shape of RF pickle file: {}".format(frames.shape) )
#             to_save = {'frames':frames, 'start_time':s[3], 'end_time':s[4], 'num_frames':len(frames)}
#             with open(os.path.join(folder_name , filename + '_read.pkl'), 'wb') as f:
#                 pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

# def cleanup_rf():
#     rf_dump_path = r"C:\Temp\mmhealth_rf_dump"
#     file_list = os.listdir(rf_dump_path)
#     file_list.sort()
#     print(file_list)
#     file_list = file_list[1:] # remove adc_data_LogFile.txt from list 
#     file_list = file_list[:-1] # remove adc_data_Raw_LogFile.csv from list
#     print(file_list)
#     file_list_copy = file_list
#     list_str = []
    

#     for file in file_list_copy:
#         file = file.lstrip('adc_data_Raw_')
#         file = file.rstrip('.bin')
#         list_str.append(file)
    
#     print(list_str)
#     list_str.sort(key = int)
#     print(list_str)
#     id = str(list_str[-1])
    
#     for file in file_list:
#         print(file)
#         print(id)
#         if (str(id) in str(file)) :
#             pass
#         else:
#             os.remove(os.path.join(rf_dump_path, file))


def cleanup_rf():
    rf_dump_path = r"C:\Temp\mmhealth_rf_dump"
    file_list = os.listdir(rf_dump_path)
    file_list_sorted = natsorted(file_list, key=lambda y: y.lower())
    file_list_sorted = file_list_sorted[1:] # remove adc_data_LogFile.txt from list 
    file_list_sorted = file_list_sorted[:-1] # remove adc_data_Raw_LogFile.csv from list
    file_time = []

    for file in file_list_sorted:
        file_path = os.path.join(rf_dump_path, file)
        file_time.append(os.path.getctime(file_path) )

    file_time_arr = np.array(file_time)

    file_time_sorted = np.sort(file_time_arr)

    value = file_time_sorted[-1]
    idx, = np.where(file_time_arr == value)
    idx = idx[0]
    file_list_sorted = file_list_sorted[:idx] + file_list_sorted[idx+1:]

    for file in file_list_sorted:
        os.remove(os.path.join(rf_dump_path, file))


def check_polarized_arrays():
    input_filepath_0 = r"E:\mmhealth_data\individual_sensor_test\polarized_1_0.tiff"
    input_filepath_45 = r"E:\mmhealth_data\individual_sensor_test\polarized_1_45.tiff"
    input_filepath_90 = r"E:\mmhealth_data\individual_sensor_test\polarized_1_90.tiff"
    input_filepath_135 = r"E:\mmhealth_data\individual_sensor_test\polarized_1_135.tiff"

    imarray_0 = imageio.volread(input_filepath_0) 
    imarray_0 = imarray_0.astype("uint8")
    imarray_45 = imageio.volread(input_filepath_45) 
    imarray_45 = imarray_45.astype("uint8")
    imarray_90 = imageio.volread(input_filepath_90) 
    imarray_90 = imarray_90.astype("uint8")
    imarray_135 = imageio.volread(input_filepath_135) 
    imarray_135 = imarray_135.astype("uint8")

    if (np.array_equal(imarray_0, imarray_45, equal_nan=False) ):
        print("0 and 45 arrays equal")
    if (np.array_equal(imarray_0, imarray_90, equal_nan=False) ):
        print("0 and 90 arrays equal")
    if (np.array_equal(imarray_0, imarray_135, equal_nan=False) ):
        print("0 and 135 arrays equal")
    if (np.array_equal(imarray_45, imarray_90, equal_nan=False) ):
        print("45 and 90 arrays equal")
    if (np.array_equal(imarray_45, imarray_135, equal_nan=False) ):
        print("45 and 135 arrays equal")
    if (np.array_equal(imarray_90, imarray_135, equal_nan=False) ):
        print("90 and 135 arrays equal")

check_polarized_arrays()