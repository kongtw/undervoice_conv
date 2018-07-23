# -*- coding: utf-8 -*-
import commands
import wavsegment as wg
import glob

num = 0
for wav_name in glob.glob('./subject/*.wav'):
    wav_8k_name = './wav_tmp/' + wav_name.split('/')[-1].split('_')[0] + str('_') + str(num) + '_.wav'
    sox_cmd = 'sox -r 6k ' + wav_name + ' -r 8k ' + wav_8k_name
    (status, output) = commands.getstatusoutput(sox_cmd)
    print '%s: %s' % (wav_8k_name, output)
    wav_index = wav_name.split('/')[-1].split('.')[0]
    audio, sample_rate = wg.read_wave(wav_8k_name)
    vad = wg.webrtcvad.Vad(int(3))
    frames = wg.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    voice_bytes = wg.vad_detect(sample_rate, 30, 300, vad, frames)
    path = './subject_vad/' + wav_index + str('_') + str(num) + '_.wav'
    wg.write_wave(path, voice_bytes, sample_rate)
    num += 1