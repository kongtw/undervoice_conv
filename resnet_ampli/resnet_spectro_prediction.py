# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys

import emospctro_input
import numpy as np
import resnet_model
import tensorflow as tf
import pymysql
import getspetrobin as getbin
from PIL import Image
import random

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'spetro2', 'spetro5 or spetro2.')
tf.app.flags.DEFINE_string('mode', 'predict', 'train or eval or predict.')
tf.app.flags.DEFINE_string('predict_data_path', '/home/dell/python/undervoice_conv/bin_data/data_*.bin',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_integer('image_size', 64, 'Image side length.')
tf.app.flags.DEFINE_string('predict_dir', '/home/dell/python/undervoice_conv/resnet_colorspectrogram/predict',
                           'Directory to keep predict outputs.')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')


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
    with tf.device(dev):

        image_p = tf.placeholder(tf.float32, [1, 64, 64, 3])
        label_p = tf.placeholder(tf.int32, [1, 5])
        model = resnet_model.ResNet(hps, image_p, label_p, FLAGS.mode)
        model.build_graph()

        label_act = np.array([[0, 0, 0, 0, 1]])
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver.restore(sess, '/home/dell/python/undervoice_conv/resnet_colorspectrogram/log_root/model.ckpt-30380')

        conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                               charset='utf8')
        cursor = conn.cursor()
        sqlstr = 'SELECT specgram_name,label FROM features WHERE label=1 OR label=3 OR label=4 OR label=0 OR label=2'
        cursor.execute(sqlstr)
        spectrogram_list = list(cursor.fetchall())
        print 'begin prediction'
        wrong_num = 0
        right_num = 0
        for spectrogram in spectrogram_list:
            im = Image.open(spectrogram[0]).resize((64, 64))
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
            print 'label is %s' % spectrogram[1]
            predictions = np.argmax(predictions, axis=1)
            print 'prediction is %s' % predictions
            if int(spectrogram[1]) == predictions:
                right_num += 1
            else:
                wrong_num += 1
        print 'right_num is %s' % str(right_num)
        print 'wrong_num is %s' % str(wrong_num)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
