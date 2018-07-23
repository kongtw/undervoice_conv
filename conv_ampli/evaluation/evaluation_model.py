from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tables
import tensorflow as tf
from nets import nets_factory

slim = tf.contrib.slim

######################
# Train Directory #
######################
tf.app.flags.DEFINE_string(
    'checkpoint_dir', '../results/TRAIN_CNN_3D',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    # 'evaluation_dataset_path', '../../data/enrollment-evaluation_sample_dataset.hdf5',
    # 'Directory where checkpoints and event logs are written to.')
    'evaluation_dataset_path', '../data/eval_sample_dataset_zs.hdf5',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    # 'development_dataset_path', '../../data/development_sample_dataset_speaker.hdf5',
    # 'Directory where checkpoints and event logs are written to.')
    'development_dataset_path', '../data/dev_sample_dataset_zs_traintest.hdf5',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'enrollment_dir', '../results/Model',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'evaluation_dir', '../results/ROC',
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

# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# Load the artificial datasets.
fileh = tables.open_file(FLAGS.evaluation_dataset_path, mode='r')
fileh_development = tables.open_file(FLAGS.development_dataset_path, mode='r')
# Test
print("Evaluation data shape:", fileh.root.utterance_evaluation.shape)
print("Evaluation label shape:", fileh.root.label_evaluation.shape)

# Get the number of subjects
num_subjects_development = len(np.unique(fileh_development.root.label_train[:]))


class predict_ampli:
    def __init__(self, process_num, class_num, is_training):
        self.graph = tf.Graph()
        self.label_act = np.array([[0]])
        self.class_num = class_num
        with self.graph.as_default():
            self.is_training = tf.placeholder(tf.bool)
            model_speech_fn = nets_factory.get_network_fn(
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
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0 / (process_num * 4.0))
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options),
                               graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                coord = tf.train.Coordinator()
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(tf.local_variables_initializer())
                latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.checkpoint_dir)
                self.saver.restore(self.sess, latest_checkpoint)

    def predict(self, speech, is_training):
        logit = self.sess.run(
            self.logits,
            feed_dict={self.is_training: is_training, self.batch_dynamic: 1, self.batch_speech: speech,
                       self.batch_labels: self.label_act})
        return logit


if __name__ == '__main__':
    print(tf.__version__)
    num_samples_per_epoch_test = fileh.root.label_evaluation.shape[0]
    num_batches_per_epoch_test = int(num_samples_per_epoch_test / FLAGS.batch_size)
    pa = predict_ampli(8, 2, False)
    step = 1
    # Loop over all batches
    for batch_num in range(num_batches_per_epoch_test):
        step += 1
        start_idx = batch_num * FLAGS.batch_size
        end_idx = (batch_num + 1) * FLAGS.batch_size
        speech_evaluation, label_evaluation = fileh.root.utterance_evaluation[start_idx:end_idx, :, :,
                                              :], fileh.root.label_evaluation[start_idx:end_idx]
        # Copy to match dimension
        speech_evaluation = np.transpose(speech_evaluation[None, :, :, :, :], axes=(1, 4, 2, 3, 0))
        print(speech_evaluation.shape)
        logit = pa.predict(speech_evaluation, False)
        print(logit)
