# -*- coding: utf-8 -*-
import pymysql
import commands
from PIL import Image


def wav_to_spetrogram():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice', charset='utf8')
    cursor = conn.cursor()
    cursor.execute('SELECT wav_name FROM features')
    wav_files = cursor.fetchall()
    spetrogram_path = '/home/dell/samples/wavsegment_specgram/'
    for wav_one in wav_files:
        wav_chunk_name = wav_one[0]
        spetrogram_name = spetrogram_path + wav_chunk_name.split('/')[-1].split('.')[0] + '.png'
        sox_cmd = 'sox ' + wav_chunk_name + ' -n rate 8k spectrogram -x 128 -y 128 -m -h -r -o ' + spetrogram_name
        (status, output) = commands.getstatusoutput(sox_cmd)
        print(status)
        label = wav_one[0].split('/')[-1].split('_')[0]
        value_one = [(spetrogram_name, int(label), wav_chunk_name)]
        cursor.executemany("UPDATE features SET specgram_name=%s, label=%s WHERE wav_name=%s", value_one)
        conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    wav_to_spetrogram()
