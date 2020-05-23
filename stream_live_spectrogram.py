#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:06:27 2020

@author: sanchit
"""

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display 

%matplotlib tk 

# define parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32 
CHANNELS = 1
RATE = 22050

def get_stream():
    """ get audio stream object """
    # pyaudio object 
    py_aud = pyaudio.PyAudio() 
    
    # get the stream object for streaming data from microphone 
    stream = py_aud.open(
            format=FORMAT, 
            channels=CHANNELS, 
            rate=RATE, 
            input=True,
            frames_per_buffer=CHUNK
        )
    return py_aud, stream 

def compute_fft_mag(data):
    """ compute power spectrum with FFTShift of the frequencies """
    # compute FT 
    X_f = np.fft.fft(data) 
    # compute magnitude of FT 
    ft_mag = abs(X_f) 
    return np.fft.fftshift(ft_mag)

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, figsize=(15,8))

py_aud, stream = get_stream()

data = stream.read(CHUNK, exception_on_overflow=False)
data = np.frombuffer(data, dtype=np.float32)
x = np.arange(0, CHUNK)


line, = ax1.plot(x, data)
#line_ft, = ax2.plot(x, compute_fft_mag(data))

mel_spectrogram = librosa.feature.melspectrogram(data, sr=RATE, n_fft=CHUNK, hop_length=int(CHUNK / 4))
log_mel_spect = librosa.power_to_db(abs(mel_spectrogram)) 

#line_ft, = ax2.plot(x, log_mel_spect)
#librosa.display.specshow(log_mel_spect, y_axis='mel', fmax=4000, x_axis='time', ax=ax2)
librosa.display.specshow(log_mel_spect, y_axis='mel', sr=RATE, hop_length=int(CHUNK / 4), x_axis='time', ax=ax2)

# basic formatting for the axes
ax1.set_title('audio waveform')
ax1.set_xlabel('samples')
ax1.set_ylabel('amplitude')

ax2.set_title('magnitude (or, power spectrum) of the Fourier Transform')
ax2.set_xlabel('samples')
ax2.set_ylabel('power')

plt.show(block=False)

# TODO: collect frames (or, audio data) of certain time and then compute spectrogram!!! 
# x * chunk_samples = SR -> x = SR/chunk_samples -> x*t secs = number of samples in t secs
# example: http://people.csail.mit.edu/hubert/pyaudio/ 

while True:
    
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(data, dtype=np.float32)   
        
        line.set_ydata(data)
        
        mel_spectrogram = librosa.feature.melspectrogram(data, sr=RATE, n_fft=CHUNK, hop_length=int(CHUNK / 4))
        log_mel_spect = librosa.power_to_db(abs(mel_spectrogram)) 
        #librosa.display.specshow(log_mel_spect, y_axis='mel', fmax=4000, x_axis='time', ax=ax2)
        librosa.display.specshow(log_mel_spect, y_axis='mel', sr=RATE, hop_length=int(CHUNK / 4), x_axis='time', ax=ax2)

        #line_ft.set_ydata(log_mel_spect)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    except Exception as e:
        print(f"excception occured: {e}")
        break 

stream.stop_stream()
stream.close()
py_aud.terminate()