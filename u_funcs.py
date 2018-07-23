#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import wave
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import wave
from PIL import Image
from scipy import signal

def speech_word_count(filename):
    wavefile = wave.open(filename, 'r')
    wav_data = wavefile.readframes(-1)
    wav_data = np.fromstring(wav_data, 'Int16')
    b, a = signal.butter(4, [0.025, 0.5], 'band')
    sf = abs(signal.filtfilt(b, a, wav_data))
    window_size = 3
    sf_average = []
    for i in xrange(0, len(sf) - window_size):
        sf_average.extend([sum(sf[i:i + window_size]) / window_size])
    rise_index = []
    window_size = 80
    for i in xrange(0, int(len(sf) / window_size) - 2):
        if max(sf_average[i * window_size:(i + 1) * window_size]) < max(sf_average[
                                                                        ((i + 1) * window_size - int(window_size / 2)):(
                                                                                (i + 2) * window_size - int(
                                                                                    window_size / 2))]):
            rise_index.extend([1])
        else:
            rise_index.extend([0])

    pre_same = 0
    word_count = 0
    same_list = []
    for i in xrange(1, len(rise_index)):
        if rise_index[i - 1] == rise_index[i]:
            same_list.extend([rise_index[i - 1]])
            pre_same += 1
            if pre_same == 6:
                word_count += 1
            else:
                pass
        else:
            pre_same = 0
            same_list = []
    return word_count

def speech_peek_count(filename):
    wavefile = wave.open(filename, 'r')
    framerate = wavefile.getframerate()
    wav_data = wavefile.readframes(-1)
    wav_data = np.fromstring(wav_data, 'Int16')
    b, a = signal.butter(4, [0.025, 0.5], 'band')
    sf = abs(signal.filtfilt(b, a, wav_data))
    (spectrum, freqs, t, im) = plt.specgram(sf, window=np.blackman(256), NFFT=256, Fs=framerate, noverlap=128)
    plt.close('all')
    spectrum_sum = np.sum(spectrum, axis=0)
    spectrum_average = (np.sum(spectrum_sum) / len(spectrum_sum)) * np.ones(spectrum_sum.shape) * 8
    peak_count = 0
    peak_flag = False
    for i in xrange(0, len(spectrum_average)):
        if (spectrum_sum[i] > spectrum_average[i]):
            if peak_flag:
                peak_count += 1
            else:
                pass
            peak_flag = False
        else:
            peak_flag = True
    return peak_count

def speech_peek_position_value(filename):
    wavefile = wave.open(filename, 'r')
    framerate = wavefile.getframerate()
    wav_data = wavefile.readframes(-1)
    wav_data = np.fromstring(wav_data, 'Int16')
    b, a = signal.butter(4, [0.025, 0.5], 'band')
    sf = abs(signal.filtfilt(b, a, wav_data))
    (spectrum, freqs, t, im) = plt.specgram(sf, window=np.blackman(256), NFFT=256, Fs=framerate, noverlap=128)
    plt.close('all')
    spectrum_sum = np.sum(spectrum, axis=0)
    spectrum_average = (np.sum(spectrum_sum) / len(spectrum_sum)) * np.ones(spectrum_sum.shape) * 8
    peak_count = 0
    peak_flag = False
    peek_position = ''
    peek_value = ''
    for i in xrange(0, len(spectrum_average)):
        if (spectrum_sum[i] > spectrum_average[i]):
            if peak_flag:
                peak_count += 1
                peek_position = peek_position + ',' + str(i)
                peek_value = peek_value + ',' + str(spectrum_sum[i])
            else:
                pass
            peak_flag = False
        else:
            peak_flag = True
    return [peek_position, peek_value]

def speech_fft_windows_count(filename):
    wavefile = wave.open(filename, 'r')
    framerate = wavefile.getframerate()
    wav_data = wavefile.readframes(-1)
    wav_data = np.fromstring(wav_data, 'Int16')
    b, a = signal.butter(4, [0.025, 0.5], 'band')
    sf = abs(signal.filtfilt(b, a, wav_data))
    (spectrum, freqs, t, im) = plt.specgram(sf, window=np.blackman(256), NFFT=256, Fs=framerate, noverlap=128)
    plt.close('all')
    spectrum_sum = np.sum(spectrum, axis=0)
    spectrum_average = (np.sum(spectrum_sum) / len(spectrum_sum)) * np.ones(spectrum_sum.shape)

    signal_len = len(spectrum_average)
    voice_begin = 0
    voice_end = 0
    for i in xrange(0, len(spectrum_average)):
        if (spectrum_sum[i] > spectrum_average[i]):
            voice_begin = i
            break
        else:
            pass
    for i in xrange(0, len(spectrum_average)):
        if (spectrum_sum[signal_len - 1 - i] > spectrum_average[signal_len - 1 - i]):
            voice_end = i
            break
        else:
            pass
    voice_fft_windows_count = (signal_len - voice_end) - voice_begin + 1
    return voice_fft_windows_count

def speech_fft_samples_count(filename):
    wavefile = wave.open(filename, 'r')
    framerate = wavefile.getframerate()
    wav_data = wavefile.readframes(-1)
    wav_data = np.fromstring(wav_data, 'Int16')
    b, a = signal.butter(4, [0.025, 0.5], 'band')
    sf = abs(signal.filtfilt(b, a, wav_data))
    (spectrum, freqs, t, im) = plt.specgram(sf, window=np.blackman(256), NFFT=256, Fs=framerate, noverlap=128)
    plt.close('all')
    spectrum_sum = np.sum(spectrum, axis=0)
    spectrum_average = (np.sum(spectrum_sum) / len(spectrum_sum)) * np.ones(spectrum_sum.shape)
    signal_len = len(spectrum_average)
    voice_begin = 0
    voice_end = 0
    for i in xrange(0, len(spectrum_average)):
        if (spectrum_sum[i] > spectrum_average[i]):
            voice_begin = i
            break
        else:
            pass
    for i in xrange(0, len(spectrum_average)):
        if (spectrum_sum[signal_len - 1 - i] > spectrum_average[signal_len - 1 - i]):
            voice_end = i
            break
        else:
            pass
    voice_fft_samples_count = ((signal_len - voice_end) - voice_begin + 1)*128
    return voice_fft_samples_count