# -*- coding: utf-8 -*-
# !/usr/bin/env python2.7
import pymysql
import ufuncs as uf
import json

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice', charset='utf8')
cursor = conn.cursor()
sqlstr = 'SELECT wav_chunk_name FROM wav_file'
cursor.execute(sqlstr)
wav_chunk_list = cursor.fetchall()
chunk_counter = 0
for wav_chunk_str in wav_chunk_list:
    wav_chunk = wav_chunk_str[0].split(',')
    for wav_chunk_one in wav_chunk:
        if wav_chunk_one != '':
            chunk_counter += 1
            swc = int(uf.speech_word_count(wav_chunk_one))
            spc = int(uf.speech_peek_count(wav_chunk_one))
            ssc = int(uf.speech_fft_samples_count(wav_chunk_one))
            [spp, spv] = uf.speech_peek_position_value(wav_chunk_one)
            [mfcc_avg, mfcc_max, mfcc_min, mfcc_std] = uf.speech_mfcc(wav_chunk_one)
            mfcc_avg_json = json.dumps(mfcc_avg.tolist())
            mfcc_max_json = json.dumps(mfcc_max.tolist())
            mfcc_min_json = json.dumps(mfcc_min.tolist())
            mfcc_std_json = json.dumps(mfcc_std.tolist())
            value_one = [(wav_chunk_one, swc, spc, spp, spv, ssc, '', mfcc_avg_json, mfcc_max_json, mfcc_min_json,
                          mfcc_std_json)]
            print(chunk_counter)
            cursor.executemany("INSERT INTO features VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", value_one)
            conn.commit()
        else:
            pass
cursor.close()
conn.close()
