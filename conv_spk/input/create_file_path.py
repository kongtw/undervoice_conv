import glob
import os

file_path = os.path.expanduser('./file_path.txt')

with open(file_path, 'w') as file_path_h:
    class_label = 0
    for sound_one in glob.glob('./Audio/subject_vad/*.wav'):
        line_w = str(class_label) + ' subject_vad/' + sound_one.split('/')[-1] + '\n'
        file_path_h.write(line_w)
        class_label += 1