# -*- coding: utf-8 -*-
import glob
import os

file_path = os.path.expanduser('./file_path.txt')

with open(file_path, 'w') as file_path_h:
    for sound_one in glob.glob('./Audio/subject_vad/*.wav'):
        if sound_one.split('/')[-1].split('_')[0] == '平淡':
            class_label = 0
        else:
            class_label = 1
        line_w = str(class_label) + ' subject_vad/' + sound_one.split('/')[-1] + '\n'
        file_path_h.write(line_w)