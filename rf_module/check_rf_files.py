from ast import parse
import os
import organizer_copy as org
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.sparse import spdiags
import heartpy as hp

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volunteerID",  default='1', type=str, help="")
    parser.add_argument("--gt_method", default='robust_fft', type=str, help="Mehtod for Ground Truth pulse rate estimation")
    parser.add_argument("--window_size",  default=11, type=int, help="Windowing size for extracting the range info")
    parser.add_argument("--bpm",  action="store_true", help="Are the frequencies in beats-per-minute or Hz")
    parser.add_argument("--gt_fs",  default=30, type=float, help="Windowing size for extracting the range info")
    parser.add_argument("--lower_freq",  default=50, type=float, help="Windowing size for extracting the range info")
    parser.add_argument("--upper_freq",  default=150, type=float, help="Windowing size for extracting the range info")
    parser.add_argument("--frame_t",  default=0.00833333333333333333333333333333, type=float, help="Frame periodicity")
    # rf_params = (3.60072e9, 60.012e12, 5e6, 256, 0.0083333335)
    # (bandwidth, slope_freq Mhz/micro-sec, sampling_freq samples/second, samples/chirp, frame periodicity)
    # parser.add_argument("--bandwidth",  default=11, type=int, help="Windowing size for extracting the range info")
    # parser.add_argument("--freq_slope",  default=11, type=int, help="Windowing size for extracting the range info")
    # parser.add_argument("--samp_f",  default=11, type=int, help="Windowing size for extracting the range info")
    # parser.add_argument("--samples",  default=11, type=int, help="Windowing size for extracting the range info")
    return parser.parse_args()

def create_fast_slow_matrix(data):
    # Taking only 1 TX and RX for now
    data_ = data[:,0,0,:]
    # DC Compensation
    data_ = (data_.T - np.mean(data_, axis = 1)).T
    data_f = np.fft.fft(data_, axis = 1)
    return data_f

def find_range(data_f, samp_f, freq_slope, samples, max_range_allowed= 1):
    # max_rang in meters
    max_idx = max_range_allowed / (samp_f * 2.98e8 / freq_slope / 2 /samples)
    data_f = data_f[:,0:int(max_idx)]
    data_f = np.abs(data_f)
    data_f = np.sum(data_f, axis=0)
    index = np.argmax(data_f)
    return index

def vibration_fft_windowing(data_f, range, args):
    data_phase = np.angle(data_f)
    data_phase = np.unwrap(data_phase, axis = 0)
    window = np.blackman(args.window_size)
    data_phase = data_phase[:, range-len(window)//2:range+len(window)//2 + 1] * window
    range = len(window)//2
    data_phase -= np.mean(data_phase, axis=0)
    phase_f = np.fft.fft(data_phase, axis = 0)
    return phase_f, range

def custom_detrend(signal, Lambda):
    '''
    custom_detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.

    Inputs
      signal          : The signal where you want to remove the trend. (1d numpy array)
      Lambda          : The smoothing parameter. (int)
    
    Output
      filtered_signal : The detrended signal. (1d numpy array)
    
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    '''
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal

def pulseRateFromPowerSpectralDensity(BVP, fs, lower_cutoff_bpm=45, upper_cutoff_bpm=150, butter_order=3, detrend=False, FResBPM = 0.01):
    '''
    Estimates a pulse rate from a BVP signal
    
    Inputs
        BVP              : A BVP timeseries. (1d numpy array)
        fs               : The sample rate of the BVP time series (Hz/fps). (int)
        lower_cutoff_bpm : The lower limit for pulse rate (bpm). (int)
        upper_cutoff_bpm : The upper limit for pulse rate (bpm). (int)
        butter_order     : Order of the Butterworth Filter. (int)
        detrend          : Detrend the input signal. (bool)
        FResBPM          : Resolution (bpm) of bins in power spectrum used to determine pulse rate and SNR. (float)
    
    Outputs
        pulse_rate       : The estimated pulse rate in BPM. (float)
    
    Daniel McDuff, Ethan Blackford, January 2019
    Copyright (c)
    Licensed under the MIT License and the RAIL AI License.
    '''
    N = (60*fs)/FResBPM

    # Detrending + nth order butterworth + periodogram
    if detrend:
        BVP = custom_detrend(np.cumsum(BVP), 100)
    if butter_order:
        [b, a] = sig.butter(butter_order, [lower_cutoff_bpm/60, upper_cutoff_bpm/60], btype='bandpass', fs = fs)
    
    BVP = sig.filtfilt(b, a, np.double(BVP))

    # Calculate the PSD and the mask for the desired range
    if detrend:
        F, Pxx = sig.periodogram(x=BVP,  nfft=N, fs=fs, detrend=False);  
    else:
        F, Pxx = sig.periodogram(x=BVP, window=np.hanning(len(BVP)), nfft=N, fs=fs)
    FMask = (F >= (lower_cutoff_bpm/60)) & (F <= (upper_cutoff_bpm/60))
    
    # Calculate predicted pulse rate:
    FRange = F * FMask
    PRange = Pxx * FMask
    MaxInd = np.argmax(PRange)
    pulse_rate_freq = FRange[MaxInd]
    pulse_rate = pulse_rate_freq*60

    return pulse_rate

def getVitals(phase_f, args):
    '''
    Get the respiratory and the heart rate from the Radar signal
    '''
    #
    best_range_phase = np.mean(phase_f,axis=1)
    best_range_phase_heart = np.copy(best_range_phase)
    half_fft = np.abs(best_range_phase[0:int(best_range_phase.shape[0]/2)-1])
    freqs = np.arange(0, 1/args.frame_t/2-0.05, step = 1/args.frame_t/2/len(half_fft))

    # Respiratory frequency
    resp_idx = np.argmax(half_fft)

    # Find lower bandpass
    if(args.bpm):
        freqs *= 60
    check_freqs = np.abs(freqs - 150)
    idx_max = check_freqs.argmin()
    freqs = freqs[0:idx_max]
    half_fft = half_fft[0:idx_max]
    half_fft_heart = np.copy(half_fft)
    
    ###########
    # Primer
    primer_temporal_signal = np.real(np.fft.ifft(best_range_phase_heart))
    primer_heart_rate = pulseRateFromPowerSpectralDensity(primer_temporal_signal, 1/args.frame_t)
    print(f"#### Primer Heart Rate : {primer_heart_rate} ####")
    #############
    
    # Find bandpass ranges 50-150bpm
    lower_bound = int(50/(1/args.frame_t/2*60)*len(phase_f)/2)
    upper_bound = int(150/(1/args.frame_t/2*60)*len(phase_f)/2)
    half_fft_heart[0:lower_bound] = 0
    half_fft_heart[upper_bound:] = 0
    best_range_phase_heart[0:lower_bound] = 0
    best_range_phase_heart[len(best_range_phase_heart)-lower_bound:] = 0
    best_range_phase_heart[upper_bound:int(len(best_range_phase_heart)/2) + (int(len(best_range_phase_heart)/2) - upper_bound)] = 0

    # Best Heart Rate without primer
    heart_idx = np.argmax(half_fft_heart)

    # Finds the max in heart rates
    heart_sort = np.argsort(half_fft_heart)

    resp_wave = np.real(np.fft.ifft(best_range_phase))
    heart_wave = np.real(np.clip(np.fft.ifft(best_range_phase_heart), -0.6, 0.6))#+1.5)*1e8

    return heart_wave, resp_wave, freqs, freqs[heart_sort][::-1], primer_heart_rate

def checkAllRFFiles(main_folder, args):
    # Get the ground truth data
    gt_vital = np.load(f"{main_folder}/vital_matrix.npy")
    # Read the raw RF data into a readable format
    file_root = f"{main_folder}/rf.pkl"
    f = open(file_root,'rb')
    s = pickle.load(f)

    N = 256
    o = org.Organizer(s, 1, 1, 1, 2*N)
    frames = o.organize()
    frames = frames[:,:,:,0::2] #remove zeros

    # (bandwidth, slope_freq Mhz/micro-sec, sampling_freq samples/second, samples/chirp, frame periodicity)
    rf_params = (3.60072e9, 60.012e12, 5e6, 256, 0.0083333335)
    bandwidth, freq_slope, samp_f, samples, frame_t = rf_params

    # Process the RF data
    data_f = create_fast_slow_matrix(frames)
    max_index = find_range(data_f, samp_f, freq_slope, samples)
    phase_f, _ = vibration_fft_windowing(data_f, max_index, args)
    heart_wave, resp_wave, freqs, top_heart_rates, primer_heart_rate = getVitals(phase_f, args)

    # Method for estimating the ground truth pulse rate
    if args.gt_method.lower() == 'robust_fft':
        gt_primer = pulseRateFromPowerSpectralDensity(gt_vital[0,:,0], args.gt_fs)
        fft_gt = np.abs(np.fft.fft(gt_vital[0,:,0]))[1:]
        gt_bpm = (np.argmax(fft_gt[0:len(fft_gt)//2]) + 1)/len(fft_gt) * args.gt_fs * 60
        gt_bpm /= round(gt_bpm / gt_primer)
    elif args.gt_method.lower() == 'robust_hpy':
        gt_primer = pulseRateFromPowerSpectralDensity(gt_vital[0,:,0], args.gt_fs)
        _, measures = hp.process(hp.scale_data(gt_vital[0,:,0]), args.gt_fs)
        gt_bpm = measures['bpm'] 
        gt_bpm /= round(gt_bpm / gt_primer)
    elif args.gt_method == 'psd':
        gt_bpm = pulseRateFromPowerSpectralDensity(gt_vital[0,:,0], args.gt_fs)
    elif args.gt_method == 'hpy':
        _, measures = hp.process(hp.scale_data(gt_vital[0,:,0]), args.gt_fs)
        gt_bpm = measures['bpm']
    elif args.gt_method.lower() == 'fft':
        fft_gt = np.abs(np.fft.fft(gt_vital[0,:,0]))[1:]
        gt_bpm = (np.argmax(fft_gt[0:len(fft_gt)//2]) + 1)/len(fft_gt) * args.gt_fs * 60
    else:
        raise ValueError

    return gt_bpm, top_heart_rates, primer_heart_rate

if __name__ == '__main__':
    args = parseArgs()
    print(args)
    # Main Folder with Radar signals
    main_folder = 'rf_examples'

    # Make sure there are useable pkl files
    pkl_files = []
    root_list = []
    for root, dir, file in os.walk(main_folder):
        # print(root, dir, file)
        for f in file:
            if f[-4:] == '.pkl':
                root_list.append(root)
                pkl_files.append(f)

    error_1 = []
    error_5 = []

    gt_list = []
    det = []
    det_5_list = []

    # Filter by Volunteer ID
    volunteer_filter = f'{args.volunteerID}_'

    for fol in root_list:
        if volunteer_filter not in fol:
            continue
        print(f"*"*40)
        print(f"Folder : {fol}")
        print(f"*"*40)

        gt_bpm, top_heart_rates, primer_heart_rate = checkAllRFFiles(fol, args)
        
        print(f"GT : {gt_bpm} ; Primer Heart Rate : {primer_heart_rate} ; Top 5 Det : {top_heart_rates[0:5]}")

        # Top 1 Error
        det_pulse = top_heart_rates[0]
        det.append(det_pulse)
        error_1.append((gt_bpm - det_pulse)**2)

        # Top 5 error
        top_5 = np.inf
        det_5 = 0
        for hr_iter in top_heart_rates[0:5]:
            if (hr_iter - gt_bpm)**2 < top_5:
                top_5 = (hr_iter - gt_bpm)**2
                det_5 = hr_iter
        det_5_list.append(det_5)
        gt_list.append(gt_bpm)
        error_5.append(top_5)

    for i in zip(gt_list, det, det_5_list):
        print(i)

    print(f"Mean Squared Error of Top 1 : {np.sqrt(np.mean(error_1))}")
    print(f"Mean Squared Error of Top 5 : {np.sqrt(np.mean(error_5))}")
