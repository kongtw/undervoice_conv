from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import speechpy
import tensorflow as tf
import soundfile as sf

import conv_spk.enrollment.nets.nets_factory as nets_factory_spk
import conv_spk.input.input_feature_vad as in_f

from sklearn.metrics.pairwise import cosine_similarity

slim = tf.contrib.slim

import glob

######################
# Train Directory #
######################
tf.app.flags.DEFINE_string(
    'enrollment_dir_spk', '../results/Model/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'checkpoint_dir_spk', '../results/TRAIN_CNN_3D',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones_ev', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu_ev', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_boolean('online_pair_selection_ev', False,
                            'Use online pair selection.')

tf.app.flags.DEFINE_integer('worker_replicas_ev', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_task_ev', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers_ev', 8,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads_ev', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps_ev', 20,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs_ev', 10,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs_ev', 500,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task_ev', 0, 'task id of the replica running the training.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type_ev',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate_ev', 1.0, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate_ev', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing_ev', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor_ev', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay_ev', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas_ev', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate_ev', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay_ev', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_string(
    'model_speech_ev', 'cnn_speech', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size_ev', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_epochs_ev', 50, 'The number of epochs for training.')

tf.app.flags.DEFINE_integer('num_gpus_ev', 1,
                            'Number of gpus used for training. (0 or 1)')

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# Get the number of subjects
num_subjects_development = 511


class spk_ev_conv:
    def __init__(self, process_num, class_num, is_training):
        self.graph = tf.Graph()
        self.label_act = np.zeros((1))
        self.class_num = class_num
        with self.graph.as_default():
            self.is_training = tf.placeholder(tf.bool)
            model_speech_ev_fn = nets_factory_spk.get_network_fn(
                FLAGS.model_speech_ev,
                num_classes=self.class_num,
                is_training=is_training)
            self.speech = tf.placeholder(tf.float32, (20, 80, 40, 1))
            self.label = tf.placeholder(tf.int32, (1))
            self.batch_dynamic = tf.placeholder(tf.int32, ())
            margin_imp_tensor = tf.placeholder(tf.float32, ())
            self.batch_speech, self.batch_labels = tf.train.batch(
                [self.speech, self.label],
                batch_size=self.batch_dynamic,
                num_threads=FLAGS.num_preprocessing_threads_ev,
                capacity=5 * FLAGS.batch_size_ev)
            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(FLAGS.num_clones_ev):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                            features, logits, end_points_speech = model_speech_ev_fn(self.batch_speech)
                            self.features = features
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
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.checkpoint_dir_spk)
                self.saver.restore(self.sess, latest_checkpoint)

    def ev_kf(self, speech, is_training):
        features = self.sess.run(
            self.features,
            feed_dict={self.is_training: is_training, self.batch_dynamic: FLAGS.batch_size_ev, self.batch_speech: speech})
        return features

    def ev_wav(self, min_time, wav_name, transform, feature_mean, feature_std):
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
        min_frame_num = int(min_time / 0.01)
        if len(logenergy_lmfe) < min_frame_num:
            return [-1, 0]
        else:
            label = 0
            sample = {'feature': logenergy_lmfe, 'label': label}
            feature, label = transform(sample)
            feature -= feature_mean
            feature /= feature_std
            feature = np.transpose(np.array(feature)[None, :, :, :, :], axes=(1, 2, 3, 4, 0))
            feature_array = self.ev_kf(feature, False)
            kf_id = wav_name.split('/')[-1].split('.')[0]
            speaker_model = np.load(FLAGS.enrollment_dir_spk + kf_id + '.npy')
            kf_score = cosine_similarity(feature_array, speaker_model)
            return [1, kf_score]


def enroll_kf():
    if FLAGS.num_gpus_ev == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus_ev == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    er_conv = spk_ev_conv(8, 511, False)
    transform = in_f.Compose(
        [in_f.CMVN(), in_f.Feature_Cube(cube_shape=(20, 80, 40), augmentation=True), in_f.ToOutput()])
    feature_mean = np.load('../data/feature_mean.npy')
    feature_std = np.load('../data/feature_std.npy')
    with tf.device(dev):
        for kf_wav in glob.glob('../input/Audio/subject_vad/*.wav'):
            kf_socre = er_conv.ev_wav(0.8, kf_wav, transform, feature_mean, feature_std)
            print(kf_wav)
            print(kf_socre)

if __name__ == '__main__':
    enroll_kf()