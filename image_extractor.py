#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:33:37 2021

@author: jass
"""

import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)

file = "audio.wav"

# load audio file with Librosa
signal, sample_rate = librosa.load(file, sr=22050)

# WAVEFORM
# display waveform
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sample_rate, alpha=0.4)
plt.xlabel("Time (s)",fontsize=20)
plt.ylabel("Amplitude",fontsize=20)
plt.title("Waveform",fontsize=20)
plt.savefig("waveform.png",dpi=300)

# FFT -> power spectrum
# perform Fourier transform
fft = np.fft.fft(signal)

# calculate abs values on complex numbers to get magnitude
spectrum = np.abs(fft)

# create frequency variable
f = np.linspace(0, sample_rate, len(spectrum))

# plot full-spectrum
plt.figure(figsize=FIG_SIZE)
plt.plot(f[:int(len(spectrum))], spectrum, alpha=0.4)
plt.xlabel("Frequency",fontsize=20)
plt.ylabel("Magnitude",fontsize=20)
plt.title("Power spectrum",fontsize=20)
plt.savefig("full-spectrum.png",dpi=300)

# take half of the spectrum and frequency
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]

# plot half-spectrum
plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency",fontsize=20)
plt.ylabel("Magnitude",fontsize=20)
plt.title("Power spectrum",fontsize=20)
plt.savefig("half-spectrum.png",dpi=300)

# STFT -> spectrogram
hop_length = 512 # in num. of samples
n_fft = 2048 # window in num. of samples

# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/sample_rate
n_fft_duration = float(n_fft)/sample_rate

print("STFT hop length duration is: {}s".format(hop_length_duration))
print("STFT window duration is: {}s".format(n_fft_duration))

# perform stft
stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

# calculate abs values on complex numbers to get magnitude
spectrogram = np.abs(stft)

# display spectrogram
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time",fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.colorbar()
plt.title("Spectrogram",fontsize=20)
plt.savefig("spectogram.png",dpi=300)


# apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time",fontsize=20)
plt.ylabel("Frequency",fontsize=20)
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogram (dB)",fontsize=20)
plt.savefig("log_spectogram.png",dpi=300)

# MFCCs
# extract 13 MFCCs
MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# display MFCCs
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time",fontsize=20)
plt.ylabel("MFCC coefficients",fontsize=20)
plt.colorbar()
plt.title("MFCCs",fontsize=20)
plt.savefig("mfcc.png",dpi=300)
# show plots
plt.show()
