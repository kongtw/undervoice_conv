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

import resnet_colorspectrogram.emospctro_input as emospectro_input
import numpy as np
import resnet_colorspectrogram.resnet_model as resnet_model
import tensorflow as tf
import pymysql
import getspetrobin as getbin
from PIL import Image
import random
from sklearn.cluster import KMeans

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'spetro5', 'spetro5 or spetro2.')
tf.app.flags.DEFINE_string('mode', 'predict', 'train or eval or predict.')
tf.app.flags.DEFINE_string('predict_data_path', '/home/dell/python/undervoice_conv/bin_data/data_*.bin',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_integer('image_size', 64, 'Image side length.')
tf.app.flags.DEFINE_string('predict_dir', '/home/dell/python/undervoice_conv/resnet_colorspectrogram/predict',
                           'Directory to keep predict outputs.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')


def main_chunk():
    test_data_path = '/home/dell/runtimedata/test_data'
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice', charset='utf8')
    cursor = conn.cursor()
    wav_tmp_path = '/home/dell/runtimedata/test_data_tmp/'
    chunk_tmp = '/home/dell/runtimedata/wav_chunk/'
    for wav_file in glob.glob(test_data_path + '/*.wav'):
        print wav_file
        gchunck.get_wavchunk(wav_file, cursor, conn, wav_tmp_path, chunk_tmp)
    cursor.close()
    conn.close()


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    batch_size = 1
    if FLAGS.dataset == 'spetro5':
        num_classes = 5
    elif FLAGS.dataset == 'spetro2':
        num_classes = 2
    hps = resnet_model.HParams(batch_size=batch_size,
                               num_classes=num_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom')

    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice', charset='utf8')
    cursor = conn.cursor()
    sql_str = 'select wav_name,chunk_name,first_chunk from runtime_wav order by wav_name,first_chunk,chunk_name;'
    cursor.execute(sql_str)
    chunk_name_list = cursor.fetchall()
    with tf.device(dev):

        image_p = tf.placeholder(tf.float32, [1, 64, 64, 3])
        label_p = tf.placeholder(tf.int32, [1, 5])
        model = resnet_model.ResNet(hps, image_p, label_p, FLAGS.mode)
        model.build_graph()

        label_act = np.array([[0, 0, 0, 0, 1]])
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver.restore(sess,
                      '/home/dell/python/undervoice_conv/resnet_colorspectrogram/log_root/model.ckpt-30380')
        print 'begin prediction'
        chunk_collection = [chunk_name_list[0][1]]
        chunk_left = chunk_name_list[1:]
        chunk_num = len(chunk_left)

        for chunk_index, chunk_name in enumerate(chunk_left):
            if chunk_name[2] == 0 or chunk_index == chunk_num - 1:
                if chunk_index == chunk_num - 1:
                    chunk_collection.extend([chunk_name])
                else:
                    pass
                speach_num_c = []
                speach_pp_c = []
                speach_pv_c = []
                speach_pn_c = []
                speach_sample_c = []
                spetrogram_name_c = []
                mfcc_avg_c = []
                mfcc_max_c = []
                mfcc_min_c = []
                mfcc_std_c = []
                mfcc_feature_c = []
                emotion_c = []
                speaker_c = []

                for chunk_name_c in chunk_collection:
                    # 录音中的字数
                    speach_num = uf.speech_word_count(chunk_name_c)
                    speach_num_c.extend([speach_num])
                    # 检测到的录音中的尖峰的位置和数值
                    [speach_peak_position, speach_peak_value] = uf.speech_peek_position_value(chunk_name_c)
                    speach_pp_json = json.dumps(speach_peak_position)
                    speach_pp_c.extend([speach_pp_json])
                    speach_pv_json = json.dumps(speach_peak_value)
                    speach_pv_c.extend([speach_pv_json])
                    # 检测到的录音中的尖峰数量
                    speach_p_num = uf.speech_peek_count(chunk_name_c)
                    speach_pn_c.extend([speach_p_num])
                    # 录音信号段的采样个数
                    sample_num = uf.speech_fft_samples_count(chunk_name_c)
                    speach_sample_c.extend([sample_num])
                    # 语谱图
                    spetrogram_name = '/home/dell/runtimedata/spec/' + chunk_name_c.split('/')[-1].split('.')[
                        0] + '.png'
                    spetrogram_name_c.extend([spetrogram_name])
                    sox_cmd = 'sox ' + chunk_name_c + ' -n rate 8k spectrogram -x 128 -y 128 -m -h -r -o ' + spetrogram_name
                    (status, output) = commands.getstatusoutput(sox_cmd)
                    print 'sox status is %s' % str(status)
                    [mfcc_avg, mfcc_max, mfcc_min, mfcc_std] = uf.speech_mfcc(chunk_name_c)
                    # mfcc平均值
                    mfcc_avg_json = json.dumps(mfcc_avg.tolist())
                    mfcc_avg_c.extend([mfcc_avg_json])
                    # mfcc最大值
                    mfcc_max_json = json.dumps(mfcc_max.tolist())
                    mfcc_max_c.extend([mfcc_max_json])
                    # mfcc最小值
                    mfcc_min_json = json.dumps(mfcc_min.tolist())
                    mfcc_min_c.extend([mfcc_min_json])
                    # mfcc标准差
                    mfcc_std_json = json.dumps(mfcc_std.tolist())
                    mfcc_std_c.extend([mfcc_std_json])
                    mfcc_feature = []
                    mfcc_feature.extend(mfcc_avg)
                    mfcc_feature.extend(mfcc_max)
                    mfcc_feature.extend(mfcc_min)
                    mfcc_feature.extend(mfcc_std)
                    mfcc_feature_c.extend([mfcc_feature])

                    im = Image.open(spetrogram_name).resize((64, 64))
                    im = np.array(im)
                    im_3 = np.ones([1, 64, 64, 3])
                    image_mean = np.mean(im)
                    image_std = np.std(im)
                    image_nor = max(image_std, 1.0 / np.sqrt(64 * 64 * 3))
                    im_3[0, :, :, 0] = (im - image_mean) / image_nor
                    im_3[0, :, :, 1] = (im - image_mean) / image_nor
                    im_3[0, :, :, 2] = (im - image_mean) / image_nor
                    (predictions) = sess.run(
                        model.predictions, feed_dict={image_p: im_3, label_p: label_act})
                    # 录音的情绪类别
                    predictions = np.argmax(predictions, axis=1)
                    print 'prediction is %s' % predictions
                    emotion_c.extend([predictions[0]])

                kmeans = KMeans(n_clusters=2, random_state=0).fit(mfcc_feature_c)
                speaker_c.extend([0])

                for i in xrange(1, len(speaker_c)):
                    if kmeans[i] == kmeans[0]:
                        speaker_c.extend([0])
                    else:
                        speaker_c.extend([1])

                for i in xrange(0, len(speaker_c)):
                    print chunk_collection[i]
                    value_one = [
                        (chunk_collection[i], str(speach_num_c[i]), str(speach_pn_c[i]), speach_pp_c[i], speach_pv_c[i],
                         str(speach_sample_c[i]), spetrogram_name_c[i], mfcc_avg_c[i], mfcc_max_c[i], mfcc_min_c[i],
                         mfcc_std_c[i], str(emotion_c[i]), speaker_c[i])]
                    cursor.executemany(
                        "INSERT INTO runtime_features VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        value_one)
                    conn.commit()
                chunk_collection = []
                chunk_collection.extend([chunk_name[1]])
            else:
                chunk_collection.extend([chunk_name[1]])
    cursor.close()
    conn.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
