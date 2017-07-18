import os
import scipy.misc
import numpy as np

from model import DeConvNET
from utils import pp

import tensorflow as tf
import logging

flags = tf.app.flags
flags.DEFINE_integer(   'epoch', 100000, 'Epoch to train [100]')
flags.DEFINE_integer(   'num_D_train', 1, 'number to train discriminator [1]')
flags.DEFINE_integer(   'num_G_train', 1, 'number to train generator [1]')
flags.DEFINE_integer(   'lr_decay_steps', 100, 'number of epochs to decay lr [100]')
flags.DEFINE_integer(   'batch_size', 24, 'The size of batch images [24]')
flags.DEFINE_integer(   'image_size', 224, 'The size of image to use (will be center cropped) [224]')
flags.DEFINE_integer(   'c_dim', 3, 'Dimension of image color. [3]')
flags.DEFINE_integer(   'gf_dim', 64, 'Number of conv in the first layer generator. [64]')
flags.DEFINE_integer(   'df_dim', 64, 'Number of conv in the first layer discriminator. [64]')
flags.DEFINE_integer(   'dfc_dim', 1024, 'Dimension of discriminator units for fully connected layer. [1024]')
flags.DEFINE_integer(   'noise_dim', 100, 'Dimension of noise. [100]')
flags.DEFINE_integer(   'train_size', np.inf, 'The size of train images [np.inf]')
flags.DEFINE_integer(   'pool_max', 20, 'The size of image pool [50]')

flags.DEFINE_float(     'learning_rate', 2e-4, 'Learning rate of for adam [0.0002]')
flags.DEFINE_float(     'lr_d', 2e-4, 'Learning rate of for adam [0.0002]')
flags.DEFINE_float(     'lr_g', 2e-4, 'Learning rate of for adam [0.0002]')
flags.DEFINE_float(     'lr_decay_rate', 0.5, 'Learning rate decay rate [0.5]')
flags.DEFINE_float(     'beta1', 0.9, 'Momentum term of adam [0.9]')
flags.DEFINE_float(     'dr_rate', 0.5, 'dropout rate [0.5]')
flags.DEFINE_float(     'KL_coeff', 2.0, 'KL divergence coefficient [2.0]')
flags.DEFINE_float(     'clip_lambda', 1.0, 'grad norm clip coefficient [1.0]')
flags.DEFINE_float(     'l1_lambda', 10., 'lambda coefficient for cycle loss [10.0]')

flags.DEFINE_string(    'root_dir', None, 'Absolute root path to save ..[]')
flags.DEFINE_string(    'save_abs_path', '/data2/junyonglee/text_to_image/saves', 'Absolute root path to save the checkpoints [[PATH]/..]')
flags.DEFINE_string(    'model_name', 'bsim', 'model name')
flags.DEFINE_string(    'dataset', 'CUB', 'The name of dataset [CUB]')
flags.DEFINE_string(    'checkpoint_dir', 'checkpoint', 'Directory name to save the checkpoints [checkpoint]')
flags.DEFINE_string(    'log_dir', 'logs', 'Directory name to save the log [logs]')
flags.DEFINE_string(    'sample_dir', 'samples', 'Directory name to save the image samples [samples]')
flags.DEFINE_string(    'crop_mode', 'random_bbox', 'mode for cropping [[random_bbox], random, center]')
flags.DEFINE_string(    'loss_mode', 'CLS', 'mode for loss [[CLS], WAS, CLS_CONTENT, CLS_INT, CLS_CONTENT_INT]')
flags.DEFINE_string(    'optim', 'Adam', 'mode for optimizer [[Adam], RMS]')
flags.DEFINE_string(    'criterionGAN', 'mse', 'criterion for GAN [[mse], abs, sce]')

flags.DEFINE_boolean(   'is_train', False, 'True for training, False for testing [False]')
flags.DEFINE_boolean(   'is_crop', True, 'True for training, False for testing [True]')
flags.DEFINE_boolean(   'is_grayscale', False, 'True for training, False for testing [True]')
flags.DEFINE_boolean(   'is_MD', False, 'True for do minibatch discrimination [True]')
flags.DEFINE_boolean(   'is_MD_both', False, 'True for do minibatch discrimination [True]')
flags.DEFINE_boolean(   'is_CA', True, 'True for do conditioning augmentation [True]')
flags.DEFINE_boolean(   'is_clip', True, 'True for do conditioning augmentation [True]')
flags.DEFINE_boolean(   'is_skip_thoughts', True, 'True for using skip thoughts for caption embedding [True]')

FLAGS = flags.FLAGS

logger = logging.getLogger()
formatter = logging.Formatter(fmt = '[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s', datefmt = '%y-%m-%d %H:%M:%S')
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)
logger.setLevel('INFO')

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    FLAGS.root_dir = os.path.join(FLAGS.save_abs_path, FLAGS.model_name, FLAGS.dataset)
    
    FLAGS.checkpoint_dir = os.path.join(FLAGS.root_dir, FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    FLAGS.sample_dir = os.path.join(FLAGS.root_dir, FLAGS.sample_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    FLAGS.log_dir = os.path.join(FLAGS.root_dir, FLAGS.log_dir)
    
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    with tf.Session() as sess:
        logger.info('Initializing Network')
        deconvnet = DeConvNET(sess, FLAGS)

        if FLAGS.is_train:
            deconvnet.train()
        else:
            deconvnet.test()

if __name__ == '__main__':
    tf.app.run()
