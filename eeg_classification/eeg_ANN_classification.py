# package import
import scipy.io as sio # .mat 파일 로드할 떄 사용
import os
import numpy as np


# load eeg

# set eeg directory
cwd = os.getcwd() # define current working directory
print("current working directory: "+"\n"+ cwd+"\n")

eeg_dir = os.path.join(cwd, "raw_eeg") # define eeg directory
print("eeg directory: "+"\n"+ eeg_dir+"\n")

eeg_list = os.listdir(eeg_dir) # os.listdir = 해당 위치의 파일 목록을 알려주는 함수
print("eeg list:")
print(eeg_list)
print("\n")

# load eeg data
for subj_idx in range(len(eeg_list)):
    eeg_file = eeg_dir+"\\"+eeg_list[subj_idx]
    eeg_data = sio.loadmat(eeg_file) # sio.loadmat = .mat 확장자를 읽어주는 함수

    raw_eeg = eeg_data['EEG_data']
    print(raw_eeg)
