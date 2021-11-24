#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 19:28:39 2021

@author: jass
"""
#%%
import json
import os
import math
import librosa
import pandas as pd
import math


DATASET_PATH = "input_dataset"
JSON_PATH = "feature_data.json"
CSV_PATH = "feature_data.csv"
SAMPLE_RATE = 22050



def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "file_name":[],
        "mfcc": [],
        "rms": [],
        "zero_cross": [],
        "chroma": [],
        "spect":[]
    }
    
    meta_data = {
        "file_name":[],
        "segments":[]
        }
    

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # process all audio files in genre sub-dir
            for f in filenames:

		# load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                
                TRACK_DURATION = math.floor(librosa.get_duration(y = signal,sr = sample_rate))
               # measured in seconds
                if TRACK_DURATION % 2 != 0:
                    TRACK_DURATION = TRACK_DURATION - 1
                SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
                
                num_segments = int(TRACK_DURATION/2)
                
                samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
                num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
                
                meta_data["file_name"].append(f)
                meta_data["segments"].append(num_segments)
#%%                
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T
                    
                    #RMS
                    rms =librosa.feature.rms(signal, sample_rate, frame_length=n_fft, hop_length=hop_length, center=True, pad_mode="reflect")
                
                
                    #zerocrossing
                    zerocrosing = librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop_length, center=True)
                
                
                    #$% Chroma Frequency 
                    chroma = librosa.feature.chroma_cens(signal, sr = sample_rate, C=None, hop_length=hop_length, fmin=None, tuning=None, n_chroma=12, n_octaves=7, bins_per_octave=36, cqt_mode='full', window=None, norm=2, win_len_smooth=41, smoothing_window='hann')
                
                
                    #Spectral Roll off
                    spect = librosa.feature.spectral_rolloff(signal, sr= sample_rate, S=None, n_fft=n_fft, hop_length= hop_length, win_length=None, window='hann', center=True, pad_mode='reflect', freq=None, roll_percent=0.85)

                    
                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        print("{}, segment:{}".format(file_path, d+1))
                        prev_f = f
                        f = f + "({})".format(d+1)
                        print(f)
                        data["file_name"].append(f)
                        data["mfcc"].append(mfcc.tolist())
                        data['rms'].append(rms.tolist())
                        data['zero_cross'].append(zerocrosing.tolist())
                        data['chroma'].append(chroma.tolist())
                        data['spect'].append(spect.tolist())
                        f = prev_f

                    
    create_json(data, JSON_PATH)
    create_csv(data, CSV_PATH)
    create_json(meta_data,"meta_data.json")
    create_meta_csv(meta_data,"meta_data.csv")

def create_json(data, json_path):
    
    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
def create_csv(data,csv_path):
    s1 = pd.Series(data["file_name"], name="file_name")
    s2 = pd.Series(data["mfcc"], name="mfcc")
    s3 = pd.Series(data["rms"], name="rms")
    s4 = pd.Series(data["zero_cross"], name="zero_cross")
    s5 = pd.Series(data["chroma"], name="chrome")
    s6 = pd.Series(data["spect"], name="spect")
    
    df = pd.concat([s1,s2,s3,s4,s5,s6], axis = 1)
    df.to_csv(csv_path, index = False)
    
def create_meta_csv(data,csv_path):
    s1 = pd.Series(data["file_name"], name="file_name")
    s2 = pd.Series(data["segments"], name="segments")
    
    df = pd.concat([s1,s2], axis = 1)
    df.to_csv(csv_path, index = False)
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)

