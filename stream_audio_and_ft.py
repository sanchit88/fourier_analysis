#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:46:40 2020

@author: sanchit
"""
import pyaudio
import numpy as np
import matplotlib.pyplot as plt


%matplotlib tk 

# define parameters
CHUNK = 1024
FORMAT = pyaudio.paFloat32 #pyaudio.paInt16
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

# data = np.frombuffer(data, dtype=np.float32)[::2] 
# x = np.arange(0, int(CHUNK//2))

line, = ax1.plot(x, data)
line_ft, = ax2.plot(x, compute_fft_mag(data))

# basic formatting for the axes
ax1.set_title('audio waveform')
ax1.set_xlabel('samples')
ax1.set_ylabel('amplitude')

ax2.set_title('magnitude (or, power spectrum) of the Fourier Transform')
ax2.set_xlabel('samples')
ax2.set_ylabel('power')

plt.show(block=False)

while True:
    
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        data = np.frombuffer(data, dtype=np.float32)  
        #data = np.frombuffer(data, dtype=np.float32)[::2]  
        
        line.set_ydata(data)
        line_ft.set_ydata(compute_fft_mag(data))
        
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    except Exception as e:
        print(f"excception occured: {e}")
        break 

stream.stop_stream()
stream.close()
py_aud.terminate()
