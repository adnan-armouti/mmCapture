import organizer_copy as org
import pickle
# from scipy.io import savemat
import sys
import numpy as np

# file_name = '15.pkl'
# file_root = '15'

# start_num = int(sys.argv[1])
# end_num = int(sys.argv[2])
# num_files = end_num - start_num + 1

# dont_do_files = []

# for file_num in range(1, int(num_files)+1):
# for file_num in range(start_num, end_num+1):

# if file_num in dont_do_files:
#     print('skipping ', file_num)
# else:   
#     print('processing ', file_num, '  *************')

file_root = sys.argv[1]
file_name = file_root+'.pkl'
f = open('./' + file_name,'rb')
print(f)
s = pickle.load(f)

# o = org.Organizer(s, 1, 4, 2, 512)
# s = np.asarray(s)
# print(s.shape)
o = org.Organizer(s, 1, 4, 3, 512)
frames = o.organize()

print(frames.shape)

# import matplotlib.pyplot as plt
# import numpy as np

# # w = np.hanning(256)
# ff = np.fft.fft(frames[0][0][0])

# plt.plot(10*np.log10(np.abs(ff)))
# plt.show()

# savemat(file_root+'.mat',{'frames':frames, 'start_time':s[3], 'end_time':s[4]})

to_save = {'frames':frames, 'start_time':s[3], 'end_time':s[4], 'num_frames':len(frames)}

with open('./' + file_root + '_read.pkl', 'wb') as f:
    pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
