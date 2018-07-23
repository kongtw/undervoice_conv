import collections
import contextlib
import sys
import wave
import commands
import struct


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 42000)
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
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def decend_or_ascend(ring_buffer):
    if len(ring_buffer) <= 1:
        return []
    else:
        compare_list = []
        counter = 0
        f_former = ring_buffer[0]
        for f in ring_buffer:
            if counter == 0:
                pass
            else:
                f_former_energy = sum(ord(i) * ord(i) for i in f_former.bytes)
                f_energy = sum(ord(i) * ord(i) for i in f.bytes)
                if f_energy > f_former_energy:
                    compare_list.extend([1])
                else:
                    compare_list.extend([0])
                f_former = f
            counter += 1
        return compare_list


def consecutivenum(comparelist):
    if len(comparelist) > 1:
        prior_elem = comparelist[0]
        comparelist = comparelist[1:]
        consecutive = 1
        l_consecutive = 0
        lc_elem = prior_elem
        for i in comparelist:
            if prior_elem == i:
                consecutive += 1
            else:
                if consecutive > l_consecutive:
                    l_consecutive = consecutive
                    lc_elem = prior_elem
                else:
                    pass
                consecutive = 1
            prior_elem = i
        if consecutive > l_consecutive:
            l_consecutive = consecutive
            lc_elem = prior_elem
        return [l_consecutive, lc_elem]
    else:
        return []


def maxnum(comparelist):
    if len(comparelist) > 1:
        mx = sum(comparelist)
        o_num = len(comparelist) - mx
        if mx > o_num:
            return [mx, 1]
        else:
            return [o_num, 0]
    else:
        return []


def wordnum(consecutive_list):
    if len(consecutive_list) >= 2:
        prior_elem = consecutive_list[0]
        consecutive_list = consecutive_list[1:]
        word_num = 0
        for i in consecutive_list:
            if prior_elem == i:
                pass
            else:
                word_num += 1

            prior_elem = i
        return word_num
    else:
        return 0


def wordcounter(frame_duration_ms, padding_duration_ms, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    descend_triggered = False
    detected_frames = []
    frame_former = frames[0]
    ring_buffer.append(frame_former)
    frames = frames[1:]
    compare_log = []
    for frame in frames:
        compare_log.extend(
            [1 if sum(ord(i) * ord(i) for i in frame.bytes) > sum(ord(i) * ord(i) for i in frame_former.bytes) else 0])
        if (not triggered) and (not descend_triggered):
            ring_buffer.append(frame)
            compare_list = decend_or_ascend(ring_buffer)
            print compare_list
            cn = maxnum(compare_list)
            if len(cn) != 0:
                [mx_num, mx_elem] = cn

                if mx_elem == 1 and mx_num > ring_buffer.maxlen * 0.6:
                    triggered = True
                    detected_frames.extend([1])
                else:
                    pass

                if mx_elem == 0 and mx_num > ring_buffer.maxlen * 0.6:
                    descend_triggered = True
                    detected_frames.extend([0])
                else:
                    pass
            else:
                pass

        elif triggered:
            ring_buffer.append(frame)
            compare_list = decend_or_ascend(ring_buffer)
            print compare_list
            cn = maxnum(compare_list)
            if len(cn) != 0:
                [mx_num, mx_elem] = cn
                if mx_elem == 0 and mx_num > ring_buffer.maxlen * 0.6:
                    triggered = False
                    detected_frames.extend([0])
                else:
                    detected_frames.extend([1])
            else:
                pass
        elif descend_triggered:
            ring_buffer.append(frame)
            compare_list = decend_or_ascend(ring_buffer)
            print compare_list
            cn = maxnum(compare_list)
            if len(cn) != 0:
                [mx_num, mx_elem] = cn
                if mx_elem == 1 and mx_num > ring_buffer.maxlen * 0.6:
                    descend_triggered = False
                    detected_frames.extend([1])
                else:
                    detected_frames.extend([0])
            else:
                pass
        frame_former = frame

    print '\n'
    print detected_frames
    print compare_log
    print 'the word num is %s' % wordnum(detected_frames)


def getwordnum(wav_name):
    audio, sample_rate = read_wave(wav_name)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    wordcounter(30, 300, frames)


if __name__ == "__main__":
    getwordnum('/home/dell/samples/wavsegments/0_0_chunk_voice-00.wav')
