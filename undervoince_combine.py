#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# 作为处理长段录音的入口
import pymysql.cursors
import glob
import wavsegment as wg
import commands
import os
import pymysql
import ufuncs as uf
import json
import resnet_colorspectrogram
import getwavchunk as gchunck
import json
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
import numpy as np
import matplotlib.pylab as plt
import time
import six
import sys


def getglobalstatics():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                           charset='utf8')
    cursor = conn.cursor()
    sqlstr = 'select speech_rate,samples_num from runtime_features'
    cursor.execute(sqlstr)
    speech_num_list = cursor.fetchall()
    speech_rate_list = []
    for i in xrange(0, len(speech_num_list)):
        speech_rate_list.extend([float(speech_num_list[i][0]) / speech_num_list[i][1]])
    speech_rate_array = np.array(speech_rate_list)
    speech_avg = np.mean(speech_rate_array)
    speech_percentile = np.percentile(speech_rate_array, 80)

    value_one = [(str(speech_avg), str(speech_percentile))]
    cursor.executemany(
        "INSERT INTO global_statics VALUES (%s, %s)", value_one)
    conn.commit()

    cursor.close()
    conn.close()


def getglobalbadscore(taskkey):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                           charset='utf8')
    cursor = conn.cursor()
    sqlstr = 'select wav_name, sum(bad_emotion_flag) + sum(bad_rate_flag) + sum(bad_disgust_flag) + sum(bad_peek_flag)  ' \
             'as bad_score from runtime_wav where taskkey = "%s" group by wav_name;' % taskkey
    cursor.execute(sqlstr)
    name_score_list = cursor.fetchall()
    score_list = []
    for elem in name_score_list:
        score_list.extend([int(elem[1])])
    score_percentile = np.percentile(score_list, 90)
    value_one = [(str(score_percentile))]
    cursor.executemany(
        'UPDATE global_statics SET bad_score=%s', value_one)
    conn.commit()
    cursor.close()
    conn.close()


def set_flag(q, taskkey, return_dict, speech_rate_statics):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                           charset='utf8')
    cursor = conn.cursor()
    """
    sqlstr = 'select speech_rate_avg,speech_rate_p from global_statics'
    cursor.execute(sqlstr)
    speech_rate_statics = cursor.fetchall()
    """

    """
    sqlstr = 'select distinct wav_name from runtime_wav where taskkey = "%s"' % taskkey
    cursor.execute(sqlstr)
    wav_name_list = cursor.fetchall()
    """
    """
    for wav_name in wav_name_list:
    """
    while not q.empty():
        wav_name = q.get(False)
        sqlstr = 'select chunk_name from runtime_wav where wav_name = "%s" and taskkey = "%s"' % (wav_name[0], taskkey)
        cursor.execute(sqlstr)
        chunk_name_list = cursor.fetchall()
        if len(chunk_name_list) != 0:
            speech_rate_index = []
            peak_index = []
            emotion_index = []
            disgust_index = []
            for chunk_name in chunk_name_list:
                sql_str = 'select wav_name, speech_rate, peak_num, samples_num, emotion_state, emotion_score, ampli_state, ampli_score from runtime_features where wav_name = "%s" and taskkey = "%s";' % (
                chunk_name[0], taskkey)
                cursor.execute(sql_str)
                feature_list = cursor.fetchall()
                print chunk_name
                if float(feature_list[0][1]) / feature_list[0][3] > speech_rate_statics[0][1]:
                    speech_rate_index.extend([1])
                else:
                    speech_rate_index.extend([0])

                if int(feature_list[0][2]) != 0:
                    peak_index.extend([1])
                else:
                    peak_index.extend([0])

                if int(feature_list[0][4]) == 1 and float(feature_list[0][5]) > 0.80:
                    emotion_index.extend([1])
                else:
                    emotion_index.extend([0])

                if int(feature_list[0][6]) == 3 and float(feature_list[0][7]) > 0.9:
                    disgust_index.extend([1])
                else:
                    disgust_index.extend([0])

            for i in xrange(0, len(speech_rate_index)):
                value_one = [(str(-1), str(speech_rate_index[i]), str(peak_index[i]), str(emotion_index[i]),
                              str(disgust_index[i]), wav_name[0], chunk_name_list[i][0])]
                try:
                    cursor.executemany(
                        'UPDATE runtime_wav SET bad_flag=%s,bad_rate_flag=%s,bad_peek_flag=%s,bad_emotion_flag=%s,bad_disgust_flag=%s'
                        ' WHERE wav_name=%s and chunk_name=%s', value_one)
                    conn.commit()
                except BaseException, e:
                    print e
                    return_dict['update_db_error'] = os.getpid()
                    return
        else:
            pass
    cursor.close()
    conn.close()


def bad_conversation_filter(taskkey, return_dict):
    print 'filtering'
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                           charset='utf8')
    cursor = conn.cursor()
    sqlstr = 'select wav_name, sum(bad_emotion_flag) + sum(bad_rate_flag) + sum(bad_disgust_flag) + sum(bad_peek_flag)  ' \
             'as bad_score from runtime_wav where taskkey = "%s" group by wav_name;' % taskkey
    cursor.execute(sqlstr)
    name_score_list = cursor.fetchall()
    score_list = []
    for elem in name_score_list:
        score_list.extend([int(elem[1])])
    score_percentile = np.percentile(score_list, 90)
    for wavname_score in name_score_list:
        wav_name = wavname_score[0]
        badscore = wavname_score[1]
        try:
            if badscore > score_percentile:
                value_one = [(str(1), str(wav_name), taskkey)]
                cursor.executemany(
                    'UPDATE runtime_wav SET bad_flag=%s WHERE wav_name=%s and taskkey=%s', value_one)
                conn.commit()
            else:
                value_one = [(str(0), str(wav_name), taskkey)]
                cursor.executemany(
                    'UPDATE runtime_wav SET bad_flag=%s WHERE wav_name=%s and taskkey=%s', value_one)
                conn.commit()
        except BaseException, e:
            print e
            return_dict['update_db_error'] = os.getpid()
            return

    cursor.close()
    conn.close()


if __name__ == '__main__':
    taskkey = '5dc61138a15f67c10509dd5ca6453179'
    return_dict = {}
    set_flag(taskkey, return_dict)
    bad_conversation_filter(taskkey, return_dict)
