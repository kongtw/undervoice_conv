# -*- coding: utf-8 -*-
import collections
import contextlib
import sys
import wave

import webrtcvad


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    # 16位采样,占2bytes,n是frame_duration_ms对应的字节数
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    unvoiced_triggered = False
    voiced_frames = []
    i = 0
    begin_index = 0
    last_index = 0
    for frame in frames:
        if (not triggered) and (not unvoiced_triggered):
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
                begin_index = i - (ring_buffer.maxlen - 1)
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                unvoiced_triggered = True
                ring_buffer.clear()
        elif triggered:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            last_index = i
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                end_index = i
                yield [b''.join([f.bytes for f in voiced_frames]), begin_index, end_index]
                ring_buffer.clear()
                voiced_frames = []
        elif unvoiced_triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                unvoiced_triggered = False
        i += 1
    if voiced_frames:
        yield [b''.join([f.bytes for f in voiced_frames]), begin_index, last_index]


def vad_allsegment_collector(sample_rate, frame_duration_ms,
                             padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    unvoiced_triggered = False
    voiced_frames = []
    unvoiced_frames = []
    segment_count = 0
    for frame in frames:
        if (not triggered) and (not unvoiced_triggered):
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                unvoiced_triggered = True
                unvoiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        elif triggered:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                segment_count += 1
                voiced_frames = []
        elif unvoiced_triggered:
            unvoiced_frames.append(frame)
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                unvoiced_triggered = False
                yield b''.join([f.bytes for f in unvoiced_frames])
                segment_count += 1
                unvoiced_frames = []
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])
    if unvoiced_frames:
        yield b''.join([f.bytes for f in unvoiced_frames])


def vad_detect(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    unvoiced_triggered = False
    voiced_frames = []
    voice_bytes = b''
    for frame in frames:
        if (not triggered) and (not unvoiced_triggered):
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                unvoiced_triggered = True
                ring_buffer.clear()
        elif triggered:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                voice_bytes += b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
        elif unvoiced_triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                unvoiced_triggered = False
    if voiced_frames:
        voice_bytes += b''.join([f.bytes for f in voiced_frames])
    return voice_bytes