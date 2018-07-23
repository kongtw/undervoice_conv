#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import pymysql
import glob
import shutil
import random


def insert_bad_samples():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                           charset='utf8')
    cursor = conn.cursor()
    dir_bad = '/home/dell/samples/bad_samples/'
    for wav_n in glob.glob(dir_bad + '*.wav'):
        print wav_n
        value_one = [(wav_n.split('/')[-1].split('.')[0], str(1))]
        cursor.executemany(
            "INSERT INTO samples_flag VALUES (%s, %s)",
            value_one)
        conn.commit()


def insert_good_samples():
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                           charset='utf8')
    cursor = conn.cursor()
    dir_init = '/home/dell/下载/测试带声纹分离的样本/'
    dir_good = '/home/dell/samples/good_samples/'
    for wav_n in glob.glob(dir_init + '*.wav'):
        print wav_n
        if random.random() < float(345) / float(1500):
            dst_name = dir_good + wav_n.split('/')[-1]
            shutil.copyfile(wav_n, dst_name)
            value_one = [(wav_n.split('/')[-1].split('.')[0], str(0))]
            cursor.executemany(
                "INSERT INTO samples_flag VALUES (%s, %s)",
                value_one)
            conn.commit()
        else:
            pass


if __name__ == '__main__':
    insert_bad_samples()
