# -*- coding: utf-8 -*-
import pymysql
import commands
from PIL import Image
import glob
import os


def wav_to_spetrogram():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice', charset='utf8')
    cursor = conn.cursor()
    wav_path = '/home/dell/samples/amplitudewav/'
    wav_re_path = '/home/dell/samples/amplitudewav_8k/'
    spetrogram_path = '/home/dell/samples/ampli_specgram/'
    counter_0 = -1
    counter_1 = -1
    for wav_one in glob.glob(wav_path + '/*'):
        label = wav_one.split('/')[-1].split('_')[0]
        if label == '4':
            label = 0
            counter_0 += 1
            wav_8k_name = wav_re_path + str(label) + '_' + str(counter_0) + '.wav'
        else:
            label = 1
            counter_1 += 1
            wav_8k_name = wav_re_path + str(label) + '_' + str(counter_1) + '.wav'
        print wav_8k_name
        sox_cmd = 'sox -r 6k ' + wav_one + ' -r 8k ' + wav_8k_name
        (status, output) = commands.getstatusoutput(sox_cmd)
        spetrogram_name = spetrogram_path + wav_8k_name.split('/')[-1].split('.')[0] + '.png'
        sox_cmd = 'sox ' + wav_8k_name + ' -n rate 8k spectrogram -x 128 -y 128 -m -h -r -o ' + spetrogram_name
        (status, output) = commands.getstatusoutput(sox_cmd)
        print(status)
        value_one = [(wav_8k_name, str(label), spetrogram_name)]
        cursor.executemany("INSERT INTO ampli_wav VALUES (%s, %s, %s)", value_one)
        conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    wav_to_spetrogram()
