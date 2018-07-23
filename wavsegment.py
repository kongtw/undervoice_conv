# -*- coding: utf-8 -*-
import collections
import commands
import contextlib
import wave
import ufuncs as uf
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


def vad_segment_c(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    unvoiced_triggered = False
    voiced_frames = []
    unvoiced_frames = []
    segment_count = 0
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
                unvoiced_frames.extend(ring_buffer)
                ring_buffer.clear()
                begin_index = i - (ring_buffer.maxlen - 1)
        elif triggered:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            last_index = i
            num_unvoiced = len([f for f in ring_buffer
                                if not vad.is_speech(f.bytes, sample_rate)])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                end_index = i - (ring_buffer.maxlen - 1)
                yield [b'1' + b''.join([f.bytes for f in voiced_frames]), begin_index, end_index]
                segment_count += 1
                voiced_frames = []
        elif unvoiced_triggered:
            unvoiced_frames.append(frame)
            ring_buffer.append(frame)
            last_index = i
            num_voiced = len([f for f in ring_buffer
                              if vad.is_speech(f.bytes, sample_rate)])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                unvoiced_triggered = False
                end_index = i - (ring_buffer.maxlen - 1)
                yield [b'0' + b''.join([f.bytes for f in unvoiced_frames]), begin_index, end_index]
                segment_count += 1
                unvoiced_frames = []
        i += 1
    if voiced_frames:
        yield [b'1' + b''.join([f.bytes for f in voiced_frames]), begin_index, last_index]
    if unvoiced_frames:
        yield [b'0' + b''.join([f.bytes for f in unvoiced_frames]), begin_index, last_index]


if __name__ == '__main__':
    wav_6k_name = '/home/dell/samples/good_samples/1611020000592095.V3.wav.SZ30877.wav'
    wav_8k_name = './wavs/test.wav'
    sox_cmd = 'sox -r 6k ' + wav_6k_name + ' -r 8k ' + wav_8k_name
    (status, output) = commands.getstatusoutput(sox_cmd)
    audio, sample_rate = read_wave(wav_8k_name)
    vad = webrtcvad.Vad(int(3))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_segment_c(sample_rate, 30, 300, vad, frames)
    chunk_tmp = './wavs/'
    for i, segment in enumerate(segments):
        path = chunk_tmp + 'chunk_voice-%004d.wav' % i
        write_wave(path, segment[0], sample_rate)
        print([segment[1], segment[2]])
        if segment[0][0] == b'0':
            print 0
        elif segment[0][0] == b'1':
            print 1
        else:
            print 'error'