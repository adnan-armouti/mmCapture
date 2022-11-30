import multiprocessing as mp
import time
import pickle
from natsort import natsorted, ns
from progressbar import progressbar

from sensors.config import *
from sensors.nir_sensor import *
from sensors.nir2_sensor import *
from sensors.thermal_sensor import *
from sensors.rf_sensor import *
import sensors.sensor
from postproc.data_interpolation import *
from postproc.tiff_to_avi import *
from postproc.check_data import *
import sensors.rf_UDP.organizer_copy as org

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

def read_pickle_rf(folder_name):
    file_list = os.listdir(folder_name)
    for file in file_list:
        filename_ext = os.path.basename(file)
        filename, ext = os.path.splitext(filename_ext)
        if (ext == ".pkl"):
            f = open(os.path.join(folder_name , filename_ext),'rb')
            s = pickle.load(f)
            o = org.Organizer(s, 1, 4, 3, 512)
            frames = o.organize()
            print("Shape of RF pickle file: {}".format(frames.shape) )
            to_save = {'frames':frames, 'start_time':s[3], 'end_time':s[4], 'num_frames':len(frames)}
            with open(os.path.join(folder_name , filename + '_read.pkl'), 'wb') as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
    

wait_time = 3 #MX800 wait time to ensure that it begins recording before any sensor does.
calibrate_mode = config.getint("mmhealth", "calibration_mode") 

def nir_main(acquisition_time, folder_name, synchronizer, verbose=False):
    save_time = config.getint("mmhealth", "save_time")
    nir_cam = NIR_Sensor(filename="nir", foldername=folder_name)
    print("Ready nir")
    synchronizer.wait()
    nir_cam.acquire_save_multiprocess(acquisition_time = acquisition_time, save_time = save_time)
    if(verbose):
        nir_cam.print_stats()

def nir2_main(acquisition_time, folder_name, synchronizer, verbose=False):
    save_time = config.getint("mmhealth", "save_time")
    nir2_cam = NIR2_Sensor(filename="nir2", foldername=folder_name)
    print("Ready nir2")
    synchronizer.wait()
    nir2_cam.acquire_save_multiprocess(acquisition_time = acquisition_time, save_time = save_time)
    if(verbose):
        nir2_cam.print_stats()

def thermal_main(acquisition_time, folder_name, synchronizer):
    save_time = config.getint("mmhealth", "save_time")
    thermal_cam = Thermal_Sensor(filename="thermal", foldername=folder_name)
    print("Ready thermal cam")
    synchronizer.wait()
    thermal_cam.acquire_save_multiprocess(acquisition_time = acquisition_time, save_time = save_time)

def rf_main(acquisition_time, folder_name, synchronizer, sensor_on=True):
    rf_s = RF_Sensor(filename="rf", foldername=folder_name, sensor_on=sensor_on)
    if(not sensor_on):
        time.sleep(120) #two minutes warm up time of the RF
    print("Ready rf device")
    # synchronizer.wait()
    rf_s.acquire(acquisition_time = acquisition_time)

def progress_main(acquistion_time, folder_name, synchronizer):
    # synchronizer.wait()
    print("\nProgress:")
    for i in progressbar(range(acquistion_time)):
        time.sleep(1)
    print("\n")

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

if __name__ == '__main__':

    #Start
    start = time.time()
    #-------------------- Sensor Config ---------------------------
    
    sensors_str = config.get("mmhealth", "sensors_list")
    sensors_list_str = aslist(sensors_str, flatten=True)

    sensors_list = []
    if(calibrate_mode == 1):
        folder_name = "calibration" + "_"
    else:
        sensors_list = [] #[progress_main]
        folder_name = str(config.getint("mmhealth", "volunteer_id") ) + "_"

    for sensor in sensors_list_str:
        if(sensor == "nir"):
            sensors_list.append(nir_main)
        elif(sensor == "nir2"):
            sensors_list.append(nir2_main)
        elif(sensor == "thermal"):
            sensors_list.append(thermal_main)
        elif(sensor == "rf"):
            sensors_list.append(rf_main)
        else:
            continue
    
    jobs = []
    print(sensors_list_str)
    num_sensors = len(sensors_list_str) #RGB, NIR, NIR2, RF
    time_acquire = config.getint("mmhealth", "acquire_time") #seconds
    sync_barrior = mp.Barrier(num_sensors)
    #-------------------- Folder Config ---------------------------
    start_num = 1
    data_folder_name = os.path.join(config.get("mmhealth", "data_path"), folder_name)

    while(os.path.exists(data_folder_name + str(start_num))):
        start_num += 1
    data_folder_name += str(start_num)
    folder_name += str(start_num)
    os.makedirs(data_folder_name)
    #-------------------- Start Sensors ----------------------------
    for sensor in sensors_list:
        proc = mp.Process(target=sensor, args= (time_acquire,folder_name,sync_barrior))
        jobs.append(proc)
        proc.start()

    for job in jobs:
        job.join() 

    end = time.time()
    print("Time taken: {}".format(end-start))
    
    #--------------------- Post-Processing ---------------------------

    # print("Cleaning up RF dump files")
    # cleanup_rf()

    # if (config.getint("mmhealth", "read_rf_pkl") == 1):
    #     print("Reading RF pickle files")
    #     read_pickle_rf(data_folder_name)
    
    # if (config.getint("mmhealth", "tiff_to_avi") == 1):
    #     print("Converting .tiff files to .avi")
    #     file_list = os.listdir(data_folder_name)
    #     for file in file_list:
    #         filename_ext = os.path.basename(file)
    #         ext = os.path.splitext(filename_ext)[1]
    #         if (ext == ".tiff"):
    #             tiff_to_avi(os.path.join(data_folder_name, file))

    #--------------------- Check Data ---------------------------
    # check_data_folder(data_folder_name)