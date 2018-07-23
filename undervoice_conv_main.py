#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
# 作为处理长段录音的入口
import commands
import glob
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np
import pymysql
import pymysql.cursors
import tensorflow as tf
from PIL import Image
from sklearn.cluster import KMeans

import getwavchunk as gchunck
import resnet_colorspectrogram.resnet_model as resnet_model
import ufuncs as uf
import undervoice_c as uvc
import wordcount_s as wcs
import conv_ampli.evaluation.nets.nets_factory as nets_factory_ampli
import conv_spk.enrollment.enrollment_uv as er_uv
import soundfile as sf
import speechpy
import conv_ampli.input.input_feature_vad as in_f
slim = tf.contrib.slim
from sklearn.metrics.pairwise import cosine_similarity

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

######################
# Train Directory #
######################
tf.app.flags.DEFINE_string(
    'checkpoint_dir_ampli', './conv_ampli/results/TRAIN_CNN_3D',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir_spk', './conv_spk/results/TRAIN_CNN_3D',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'evaluation_dataset_path_ampli', './conv_ampli/data/eval_sample_dataset_zs.hdf5',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'evaluation_dataset_path_spk', './conv_spk/data/eval_sample_dataset_zs.hdf5',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'development_dataset_path_dir_ampli', './conv_ampli/data/dev_sample_dataset_zs_traintest.hdf5',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'development_dataset_path_dir_spk', './conv_ampli/data/dev_sample_dataset_zs_traintest.hdf5',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'enrollment_dir_ampli', './conv_ampli/results/Model',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'evaluation_dir_ampli', './conv_ampli/results/ROC',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_boolean('online_pair_selection', False,
                            'Use online pair selection.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 20,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 10,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 500,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 1.0, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_string(
    'model_speech', 'cnn_speech', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_epochs', 50, 'The number of epochs for training.')


class Predict:
    def __init__(self, modelname, process_num, hps, class_num):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image_p = tf.placeholder(tf.float32, [1, 64, 64, 3])
            if class_num == 5:
                self.label_p = tf.placeholder(tf.int32, [1, 5])
            elif class_num == 2:
                self.label_p = tf.placeholder(tf.int32, [1, 2])
            else:
                pass
            if class_num == 5:
                self.label_act = np.array([[0, 0, 0, 0, 1]])
            elif class_num == 2:
                self.label_act = np.array([[0, 1]])
            else:
                pass
            self.model = resnet_model.ResNet(hps, self.image_p, self.label_p, FLAGS.mode)
            self.model.build_graph()
            self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0 / (process_num * 6.0))
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options),
                               graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, modelname)
        self.prediction_label = -1
        self.prediction_score = 0.0

    def predict(self, im_3):
        (predictions) = self.sess.run(
            self.model.predictions, feed_dict={self.image_p: im_3, self.label_p: self.label_act})
        self.prediction_label = np.argmax(predictions, axis=1)
        self.prediction_score = np.max(predictions, axis=1)
        return [self.prediction_label[0], self.prediction_score[0]]


class predict_ampli_conv:
    def __init__(self, process_num, class_num, is_training):
        self.graph = tf.Graph()
        self.label_act = np.array([[0]])
        self.class_num = class_num
        with self.graph.as_default():
            self.is_training = tf.placeholder(tf.bool)
            model_speech_fn = nets_factory_ampli.get_network_fn(
                FLAGS.model_speech,
                num_classes=self.class_num,
                is_training=is_training)
            self.speech = tf.placeholder(tf.float32, (20, 80, 40, 1))
            self.label = tf.placeholder(tf.int32, (1))
            self.batch_dynamic = tf.placeholder(tf.int32, ())
            margin_imp_tensor = tf.placeholder(tf.float32, ())
            self.batch_speech, self.batch_labels = tf.train.batch(
                [self.speech, self.label],
                batch_size=self.batch_dynamic,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(FLAGS.num_clones):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                            features, logits, end_points_speech = model_speech_fn(self.batch_speech)
                            self.logits = logits
                            label_onehot = tf.one_hot(tf.squeeze(self.batch_labels, [1]), depth=self.class_num,
                                                      axis=-1)
                            with tf.name_scope('loss'):
                                loss = tf.reduce_mean(
                                    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_onehot))
                            with tf.name_scope('accuracy'):
                                correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label_onehot, 1))
                                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            variables_to_restore = slim.get_variables_to_restore()
            self.saver = tf.train.Saver(variables_to_restore, max_to_keep=20)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0 / (process_num * 6.0))
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options),
                               graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                coord = tf.train.Coordinator()
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(tf.local_variables_initializer())
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.checkpoint_dir_ampli)
                self.saver.restore(self.sess, latest_checkpoint)

    def predict(self, speech, is_training):
        logit = self.sess.run(
            self.logits,
            feed_dict={self.is_training: is_training, self.batch_dynamic: 1, self.batch_speech: speech,
                       self.batch_labels: self.label_act})
        return logit

    def predcit_ampli_along_wav(self, frame_time, wav_name, transform, feature_mean, feature_std):
        signal, fs = sf.read(wav_name)
        ###########################
        ### Feature Extraction ####
        ###########################
        # DEFAULTS:
        num_coefficient = 40
        # Staching frames
        frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.025,
                                                  frame_stride=0.01,
                                                  zero_padding=True)
        # # Extracting power spectrum (choosing 3 seconds and elimination of DC)
        power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=2 * num_coefficient)[:, 1:]
        logenergy_lmfe = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.025, frame_stride=0.01,
                                               num_filters=num_coefficient, fft_length=1024, low_frequency=0,
                                               high_frequency=None)
        frame_sample_num = int(frame_time / 0.01)
        if len(logenergy_lmfe) < frame_sample_num:
            return -1
        else:
            steps = len(logenergy_lmfe) / frame_sample_num
            feature_str = ''
            label_str = ''
            bad_count = 0
            for i in range(0, steps):
                lmfe_chunk = logenergy_lmfe[i * frame_sample_num: (i + 1) * frame_sample_num]
                label = 0
                sample = {'feature': lmfe_chunk, 'label': label}
                feature, label = transform(sample)
                feature -= feature_mean
                feature /= feature_std
                feature = np.transpose(np.array(feature)[None, :, :, :, :], axes=(1, 2, 3, 4, 0))
                feature_array = self.predict(feature, False)
                label = np.argmax(feature_array, axis=1)
                feature_str += str(feature_array[0][label[0]]) + ','
                if label[0] == 1:
                    label_str += str(1) + ','
                    bad_count += 1
                else:
                    label_str += str(0) + ','
            return [feature_str, label_str, bad_count]


class Predict_prob:
    def __init__(self, modelname, process_num, hps, class_num):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.image_p = tf.placeholder(tf.float32, [1, 64, 64, 3])
            if class_num == 5:
                self.label_p = tf.placeholder(tf.int32, [1, 5])
            elif class_num == 2:
                self.label_p = tf.placeholder(tf.int32, [1, 2])
            else:
                pass
            if class_num == 5:
                self.label_act = np.array([[0, 0, 0, 0, 1]])
            elif class_num == 2:
                self.label_act = np.array([[0, 1]])
            else:
                pass
            self.model = resnet_model.ResNet(hps, self.image_p, self.label_p, FLAGS.mode)
            self.model.build_graph()
            self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0 / (process_num * 6.0))
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options),
                               graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, modelname)
        self.prediction = []

    def predict(self, im_3):
        (predictions) = self.sess.run(
            self.model.predictions, feed_dict={self.image_p: im_3, self.label_p: self.label_act})
        self.prediction = predictions
        return self.prediction


def main_chunk(q, hash_str, return_dict):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice', charset='utf8')
    cursor = conn.cursor()
    wav_tmp_path = '/home/dell/runtimedata/test_data_tmp/'
    chunk_tmp = '/home/dell/runtimedata/wav_chunk/'
    while not q.empty():
        wav_file = q.get()
        print '%s : %s' % (
            wav_file,
            str(gchunck.get_wavchunk_all(wav_file, cursor, conn, wav_tmp_path, chunk_tmp, hash_str, return_dict)))
        gchunck.cal_wav_snr(wav_file, cursor, conn, hash_str)
        if len(return_dict) != 0:
            return
        else:
            pass
    cursor.close()
    conn.close()


def main(_):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice', charset='utf8')
    cursor = conn.cursor()
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    hash_str = hashlib.md5(time_str).hexdigest()
    print hash_str
    process_num = 8
    manager = multiprocessing.Manager()
    q = manager.Queue()
    return_dict = manager.dict()

    test_data_path = '/home/dell/runtimedata/test_data'
    for wav_file in glob.glob(test_data_path + '/*.wav'):
        q.put(wav_file)
    p = Pool(process_num)
    for i in range(process_num):
        p.apply_async(main_chunk, args=(q, hash_str, return_dict,))
    p.close()
    p.join()
    print 'all get wav chunk subprocesses & cal snr done'

    if os.path.exists(test_data_path):
        shutil.rmtree(test_data_path)
        os.mkdir(test_data_path)
    else:
        os.mkdir(test_data_path)

    if os.path.exists('/home/dell/runtimedata/test_data_tmp/'):
        shutil.rmtree('/home/dell/runtimedata/test_data_tmp/')
        os.mkdir('/home/dell/runtimedata/test_data_tmp/')
    else:
        os.mkdir('/home/dell/runtimedata/test_data_tmp/')

    if len(return_dict) != 0:
        return -1
    else:
        pass

    sql_str = 'select distinct wav_conver_name, wav_index from runtime_prob where taskkey = "%s" and use_flag = 1 ' \
              'order by wav_conver_name;' % hash_str
    print sql_str
    cursor.execute(sql_str)
    wav_name_list = cursor.fetchall()
    chunk_counter = 0
    for wav_name in wav_name_list:
        sql_str = 'select chunk_name, wav_index, kf_id from runtime_prob where wav_conver_name = "%s" and taskkey = "%s"' \
                  ' and voice_flag = 1' \
                  ' order by chunk_name;' % (wav_name[0], hash_str)
        print sql_str
        cursor.execute(sql_str)
        chunk_tmp = cursor.fetchall()
        chunk_counter += len(chunk_tmp)
        q.put(chunk_tmp)
    print 'begin to get features'
    process_num = 1
    p = Pool(process_num)
    for i in range(process_num):
        p.apply_async(process_fun, args=(q, process_num, hash_str, return_dict,))
    p.close()
    p.join()
    print 'all get feature subprocesses done'

    uvc.filter_ampli_detail(hash_str)
    print hash_str
    cursor.close()
    conn.close()
    if len(return_dict) != 0:
        print return_dict
        return [-1, 'duplicate error']
    else:
        return [0, hash_str]


def spk_sp(chunk_name_list, cursor, spk_model_path, er_conv, transform, feature_mean, feature_std):
    kf_id = chunk_name_list[0][2]
    sql_str = 'select count(distinct kf_id) from register_kf where kf_id = "%s"' % kf_id
    cursor.execute(sql_str)
    kf_num = cursor.fetchall()
    if kf_num[0][0] == 0:
        return -1
    else:
        kf_flag_list = []
        kf_score_list = []
        ubm_score_list = []
        ubm_model = np.load(spk_model_path + 'ubm.npy')
        speaker_model = np.load(spk_model_path + kf_id + '.npy')
        for wav_name in chunk_name_list:
            feature_array = er_conv.er_wav(0.8, wav_name[0], transform, feature_mean, feature_std)
            if feature_array[0] == -1:
                kf_flag_list.extend([-1])
                kf_score_list.extend([-1])
                ubm_score_list.extend([-1])
            else:
                kf_score = cosine_similarity(feature_array[1], speaker_model)
                ubm_score = cosine_similarity(feature_array[1], ubm_model)
                print('kf score %f ' % kf_score[0][0])
                print('ubm score %f ' % ubm_score[0][0])
                if kf_score[0][0] > ubm_score[0][0]:
                    kf_flag_list.extend([1])
                else:
                    kf_flag_list.extend([0])
                kf_score_list.extend([kf_score[0][0]])
                ubm_score_list.extend([ubm_score[0][0]])

        print [kf_flag_list, kf_score_list, ubm_score_list]
        return [kf_flag_list, kf_score_list, ubm_score_list]

def process_fun(chunk_queue, process_num, hash_str, return_dict):
    print 'process id is %s' % os.getpid()
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')
    batch_size = 1
    num_classes = 5
    hps_emo = resnet_model.HParams(batch_size=batch_size,
                                   num_classes=num_classes,
                                   min_lrn_rate=0.0001,
                                   lrn_rate=0.1,
                                   num_residual_units=5,
                                   use_bottleneck=False,
                                   weight_decay_rate=0.0002,
                                   relu_leakiness=0.1,
                                   optimizer='mom')

    num_classes = 2
    hps_ampli = resnet_model.HParams(batch_size=batch_size,
                                     num_classes=num_classes,
                                     min_lrn_rate=0.0001,
                                     lrn_rate=0.1,
                                     num_residual_units=5,
                                     use_bottleneck=False,
                                     weight_decay_rate=0.0002,
                                     relu_leakiness=0.1,
                                     optimizer='mom')

    pa = predict_ampli_conv(process_num, 2, False)
    er_conv = er_uv.spk_er_conv(process_num, 511, False)
    transform = in_f.Compose(
        [in_f.CMVN(), in_f.Feature_Cube(cube_shape=(20, 80, 40), augmentation=True), in_f.ToOutput()])
    feature_mean_ampli = np.load('./conv_ampli/data/feature_mean.npy')
    feature_std_ampli = np.load('./conv_ampli/data/feature_std.npy')
    feature_mean_spk = np.load('./conv_spk/data/feature_mean.npy')
    feature_std_spk = np.load('./conv_spk/data/feature_std.npy')
    spk_model_path = './conv_spk/results/Model/'
    with tf.device(dev):

        model_emo_name = '/home/dell/python/undervoice_conv/resnet_colorspectrogram/log_root/model.ckpt-30380'
        model_ampli_name = '/home/dell/python/undervoice_conv/resnet_ampli/log_root/model.ckpt-4836'

        predict_emo = Predict_prob(model_emo_name, process_num, hps_emo, 5)
        predict_ampli = Predict_prob(model_ampli_name, process_num, hps_ampli, 2)

        print 'begin prediction'
        conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='undervoice',
                               charset='utf8')
        cursor = conn.cursor()

        while not chunk_queue.empty():
            chunk_name_list = chunk_queue.get()
            speach_num_c = []
            spetrogram_name_c = []
            mfcc_avg_c = []
            mfcc_max_c = []
            mfcc_min_c = []
            mfcc_std_c = []
            mfcc_feature_c = []
            emotion_s = []
            speaker_c = []
            kf_score = []
            ubm_score = []
            ampli_s = []
            detail_flag_list = []
            feature_str_list = []
            label_str_list = []
            bad_count_list = []
            sp_res = spk_sp(chunk_name_list, cursor, spk_model_path, er_conv, transform, feature_mean_spk, feature_std_spk)
            for chunk_name in chunk_name_list:
                chunk_name_c = chunk_name[0]
                pa_r = pa.predcit_ampli_along_wav(1.6, chunk_name_c, transform, feature_mean_ampli, feature_std_ampli)
                if pa_r == -1:
                    detail_flag_list.extend([0])
                    feature_str = '-1'
                    label_str = '-1'
                    bad_count = 0
                else:
                    feature_str, label_str, bad_count = pa_r
                    detail_flag_list.extend([1])
                feature_str_list.extend([feature_str])
                label_str_list.extend([label_str])
                bad_count_list.extend([bad_count])
                # 录音中的字数
                speach_num = wcs.getwordnum(chunk_name_c)
                speach_num_c.extend([speach_num])
                # 语谱图
                spetrogram_name = '/home/dell/runtimedata/spec/' + chunk_name_c.split('/')[-1].split('.')[0] + '.png'
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
                # 录音的情绪及态度类别
                emo_prediction_prob = predict_emo.predict(im_3)
                ampli_prediction_prob = predict_ampli.predict(im_3)
                emotion_s.extend(emo_prediction_prob)
                ampli_s.extend(ampli_prediction_prob)

            if sp_res == -1:
                if len(chunk_name_list) > 1:
                    kmeans = KMeans(n_clusters=2, random_state=0).fit(mfcc_feature_c)
                    speaker_c.extend([1])
                    distance = np.abs(np.asarray(mfcc_feature_c[0]) - kmeans.cluster_centers_[0])
                    kf_score.extend([np.sum(distance)])
                    ubm_score.extend([1000.00])
                    for i in xrange(1, len(kmeans.labels_)):
                        if kmeans.labels_[i] == kmeans.labels_[0]:
                            speaker_c.extend([1])
                            distance = np.abs(np.asarray(mfcc_feature_c[0]) - kmeans.cluster_centers_[0])
                            kf_score.extend([np.sum(distance)])
                        else:
                            speaker_c.extend([0])
                            distance = np.abs(np.asarray(mfcc_feature_c[0]) - kmeans.cluster_centers_[1])
                            kf_score.extend([np.sum(distance)])
                        ubm_score.extend([1000.00])
                else:
                    speaker_c.extend([1])
                    kf_score.extend([0.0])
                    ubm_score.extend([1000.00])
            else:
                speaker_c = sp_res[0]
                kf_score = sp_res[1]
                ubm_score = sp_res[2]

            for i in xrange(0, len(speaker_c)):
                if sp_res == -1:
                    value_one = [
                        (str(speach_num_c[i]),
                         spetrogram_name_c[i],
                         str(emotion_s[i][0]), str(emotion_s[i][1]), str(emotion_s[i][2]),
                         str(emotion_s[i][3]), str(emotion_s[i][4]), str(ampli_s[i][0]), str(ampli_s[i][1]),
                         str(speaker_c[i]), str(0), str(kf_score[i]), str(ubm_score[i]), str(detail_flag_list[i]),
                         feature_str_list[i],
                         label_str_list[i], str(bad_count_list[i]), str(chunk_name_list[i][0]),
                         hash_str)]
                else:
                    value_one = [
                        (str(speach_num_c[i]),
                         spetrogram_name_c[i],
                         str(emotion_s[i][0]), str(emotion_s[i][1]), str(emotion_s[i][2]),
                         str(emotion_s[i][3]), str(emotion_s[i][4]), str(ampli_s[i][0]), str(ampli_s[i][1]),
                         str(speaker_c[i]), str(1), str(kf_score[i]), str(ubm_score[i]), str(detail_flag_list[i]),
                         feature_str_list[i],
                         label_str_list[i], str(bad_count_list[i]), str(chunk_name_list[i][0]),
                         hash_str)]
                try:
                    cursor.executemany(
                        "UPDATE runtime_prob SET speech_word_num=%s, specgram_name=%s, es_1=%s, es_2=%s, es_3=%s, "
                        "es_4=%s, es_5=%s, as_1=%s, as_2=%s, speaker_label=%s, sp_meth=%s, kf_score=%s, ubm_score=%s, "
                        "detail_flag=%s, feature_str=%s, label_str=%s, bad_count=%s "
                        "WHERE chunk_name=%s and taskkey=%s",
                        value_one)
                    conn.commit()
                except BaseException, e:
                    print e
                    print 'error'
                    return_dict['db_insert_error'] = chunk_name_list[i][0] + ' ' + str(e)
                    print chunk_name_list[i][0]
                    exit(-1)
            for dl_one in chunk_name_list:
                os.remove(dl_one[0])
            for dl_one in spetrogram_name_c:
                os.remove(dl_one)

    cursor.close()
    conn.close()


if __name__ == '__main__':
    print(tf.__version__)
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
