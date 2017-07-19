from __future__ import division
import os
import time
import datetime
from glob import glob
from six.moves import xrange
import fnmatch

import tensorflow as tf
import numpy as np

from ops import *
from utils import *

from tqdm import trange

from logging import getLogger
logger = getLogger(__name__)

from optim_manager import OptimManager
from summary_manager import SummaryManager

from ava_manager import AVAManager

class DeConvNET(object):
    
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

        logger.info('Loading Data...')
        self.embedding_dim = 1024
        self.augment_dim = int(self.embedding_dim / 8)
        self.get_data_train = (AVAManager('/data1/AVA', config.type)).get_data
        self.get_data_val = (AVAManager('/data1/AVA', config.type, train = False)).get_data
        logger.info('Loading Data...Done!')

        self.criterion = criterion(config.criterion)
        
        self.optimManager = OptimManager(optim='Adam', beta1=self.config.beta1, is_clip=self.config.is_clip, clip_lambda=self.config.clip_lambda)
        self.summaryManager = SummaryManager()
        self.build_model()
        
    def build_model(self):
        self.build_network()
        self.build_loss()
        
        self.build_variables()

        self.build_optim()
        self.build_summary()
        
        self.build_saver(self.t_vars)
        self.build_writer()
        
    def build_network(self):
        logger.info('Initializing NETWORK...')
        self.lr = tf.placeholder(tf.float32, name = 'learning_rate')
        self.dr_rate = tf.placeholder(tf.float32, name = 'dropout_rate')
        self.bn_train_phase = tf.placeholder(tf.bool, name = 'phase')
    
        ############# input #############
        self.images = tf.placeholder(tf.float32, [self.config.batch_size] + [self.config.image_size, self.config.image_size, self.config.c_dim], name = 'images')
        self.scores_expt = tf.placeholder(tf.float32, [self.config.batch_size] + [1], name = 'scores')
    
        ########## Regressor ##########
        logger.info('Initializing Regressor ...')
        with tf.variable_scope('Regressor') as scope:
            self.scores_pred = self.regressor(self.images, bn_train_phase = self.bn_train_phase, scope = scope, reuse = False)
        logger.info('Initializing Discriminator... DONE')
        
        logger.info('Initializing NETWORK... DONE\n')
    
    def build_loss(self):
        logger.info('Initializing LOSS...')

        # regressor loss
        with tf.variable_scope('R_loss'):
            self.r_loss = self.criterion(logits = self.scores_pred, labels = self.scores_expt, name = self.summaryManager.get_sum_marked_name('3_r_sharp_loss'))
    
        logger.info('Initializing LOSS... DONE\n')
        
    def build_variables(self):
        logger.info('Initializing NETWORK VARIABLE...')
        self.t_vars = tf.trainable_variables()
        self.r_vars = [var for var in self.t_vars if 'Regressor' in var.name]
        logger.info('Initializing NETWORK VARIABLE... DONE\n')
        
        show_all_variables(self.r_vars, 'Regressor')
        show_all_variables(verbose = False)

        logger.info('Initializing SUMMARY VARIABLE...')
        self.build_sum_var()
        logger.info('Initializing SUMMARY VARIABLE... DONE\n')
        
    def build_sum_var(self):
        logger.info('Initializing SUMMARY VARIABLE...')
        with tf.variable_scope('input'):
            self.summaryManager.add_image_sum(self.images, 'input')
    
        self.summaryManager.set_sum_vars(self.sess.graph)
        logger.info('Initializing SUMMARY VARIABLE... DONE')
        
    def build_optim(self):
        ###### Optimizer for network ######
        logger.info('Initializing Optimizer ...')
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.variable_scope('Optimizer'):
                self.r_optim, self.grad_and_vars_r = self.optimManager.get_optim_and_grad_vars(self.lr, self.r_loss, self.r_vars)
        logger.info('Initializing Optimizer ... DONE\n')
    
    def build_summary(self):
        logger.info('Initializing Summary ...')
        self.itm_sum = self.summaryManager.get_merged_summary() # images, histograms
        self.r_loss_sum = self.summaryManager.get_merged_summary(self.r_vars, grad_norm = get_grads_norm(self.grad_and_vars_r), name = 'R_sum')
        logger.info('Initializing Summary ... DONE\n')
        
    def build_saver(self, vars):
        logger.info('Loading Saver...')
        self.saver_train = tf.train.Saver(vars)
        logger.info('Loading Saver... DONE\n')
        
    def build_writer(self):
        logger.info('Loading Writer...')
        self.writer = tf.summary.FileWriter(self.config.log_dir, self.sess.graph)
        logger.info('Loading Writer... Done\n')

    def train(self):
        counter = 0
        
        ##### INIT #####
        self.sess.run(tf.global_variables_initializer())
        ##### TRAIN STARTS #####
        lr = self.config.learning_rate
        # load data
        val_files, val_scores = self.get_data_val()
        
        logger.info('Training Starts!')
        it_epoch = trange(self.config.epoch, ncols = 100, initial = 0, desc = 'Epoch')
        for epoch in it_epoch:
            
            train_files, train_scores = self.get_data_train()

            batch_idxs = min(len(train_files), self.config.train_size) // self.config.batch_size
            it_train = trange(batch_idxs, ncols = 100, initial = 0, desc = '[Train]')
            for idx in it_train:
                
                batch_range = np.arange(idx * self.config.batch_size, (idx + 1) * self.config.batch_size)
                
                batch_images = get_images(train_files, batch_range, self.config.is_crop, self.config.crop_mode, self.config.image_size, None, self.config.is_grayscale)
                scores_expt = np.take(train_scores, batch_range, axis = 0).reshape((self.config.batch_size, 1))

                lr = get_learning_rate(lr, epoch, counter, self.config.lr_decay_steps, self.config.lr_decay_rate)
                # Update Regressor
                _, r_loss, summary_str = self.sess.run([self.r_optim, self.r_loss, self.r_loss_sum],
                                                         feed_dict = {
                                                             self.lr: lr, self.dr_rate: self.config.dr_rate, self.bn_train_phase: True,
                                                             self.images: batch_images, self.scores_expt : scores_expt})
                self.writer.add_summary(summary_str, counter)
                
                # Write img summary
                summary_str = self.sess.run(self.itm_sum,
                                            feed_dict = {
                                                self.dr_rate: self.config.dr_rate, self.bn_train_phase: False,
                                                self.images: batch_images, self.scores_expt: scores_expt})
                self.writer.add_summary(summary_str, counter)

                it_train.set_description(('[Train] epoch: %d, r_loss: %.4f' % (epoch, r_loss)))
                
                counter += 1
            
            # validation
            if np.mod(epoch, 1) == 0:
                val_batch_idxs = 0
                val_batch_range = np.arange(val_batch_idxs * self.config.batch_size, (val_batch_idxs+ 1) * self.config.batch_size)
    
                val_images = get_images(val_files, val_batch_range, self.config.is_crop, self.config.crop_mode, self.config.image_size, None, self.config.is_grayscale)
                val_scores= np.take(val_scores, val_batch_range, axis = 0).reshape((self.config.batch_size, 1))

                r_loss = self.sess.run(self.r_loss,
                    feed_dict = {self.dr_rate: self.config.dr_rate, self.bn_train_phase: False,
                                 self.images: val_images, self.scores_expt: val_scores})
                
                it_epoch.set_description(('[Sample] epoch: %d, r_loss: %.4f' % (epoch, r_loss)))
            # save checkpoint
            if np.mod(epoch, 1) == 0:
                self.save(self.saver_train, self.config.checkpoint_dir, self.config.model_name, counter)
    
    def test(self):
        if self.load(self.saver_train, self.config.checkpoint_dir):
            logger.info(' [*] Load SUCCESS')
        else:
            logger.info(' [!] Load failed...')

        '''
        if self.config.dataset == 'MSCOCO':
        elif self.config.dataset == 'bird':
        samples_cap, samples_img, loss = self.sess.run(
            [self.sample_from_caption_embeddings, self.sample_from_Image_embeddings, self.loss],
            feed_dict = {self.images: val_images, self.captions: val_captions}
        )
        
        timestr = time.strftime("%m/%d_%H:%M")

        output_size = np.int32(np.ceil(np.sqrt(self.config.batch_size)))
        save_images(samples_cap, [output_size, output_size],
                    '{}/train_{}_caption_{}.png'.format(self.config.sample_dir, self.config.dataset, timestr))
        save_images(samples_img, [output_size, output_size],
                    '{}/train_{}_gt_{}.png'.format(self.config.sample_dir, self.config.dataset, timestr))
        print('[Sample] loss: %.8f' % (loss))
        '''
        
    def regressor(self, input, bn_train_phase, scope = 'Regressor', reuse = False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
        
            h0 = conv2d(input, self.config.df_dim, name = 'h0_conv_0')
            h0 = lrelu(bnorm(h0, bn_train_phase, name = 'h0_bn_1'))
            h0 = tf.nn.avg_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h0_pool_2') #112
        
            h1 = conv2d(h0, self.config.df_dim * 2, name = 'h1_conv_0')
            h1 = lrelu(bnorm(h1, bn_train_phase, name = 'h1_bn_1'))
            h1 = tf.nn.avg_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h1_pool_2') #56
        
            h2 = conv2d(h1, self.config.df_dim * 4, name = 'h2_conv_0')
            h2 = lrelu(bnorm(h2, bn_train_phase, name = 'h2_bn_1'))
            h2 = tf.nn.avg_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h2_pool_2') #28
        
            h3 = conv2d(h2, self.config.df_dim * 8, name = 'h3_conv_0')
            h3 = lrelu(bnorm(h3, bn_train_phase, name = 'h3_bn_1'))
            h3 = tf.nn.avg_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h3_pool_2') #14
        
            h4 = conv2d(h3, self.config.df_dim * 8, name = 'h4_conv_0')
            h4 = lrelu(bnorm(h4, bn_train_phase, name = 'h4_bn_1'))
            image_embeddings = tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h4_pool_2') #7
        
            image_embeddings_h, image_embeddings_w = image_embeddings.get_shape().as_list()[1:3]
            h4 = lrelu(conv2d(image_embeddings, self.config.dfc_dim, k_h = image_embeddings_h, k_w = image_embeddings_w, padding = 'VALID', name = 'h4_linear_0'))
            h4 = lrelu(conv2d(h4, self.config.dfc_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h4_linear_1'))
            h4 = conv2d(h4, 1, k_h = 1, k_w = 1, padding = 'VALID', name = 'h4_linear_2')
            h4 = tf.sigmoid(h4)
            logits = tf.reshape(h4, [self.config.batch_size, -1])
        
            return logits
        
    def get_data(self, data_paths, size = None):
        filenames, captions, bboxes, filenames_wrong, bboxes_wrong = self.cubManager.get_captions(data_paths[0], data_paths[1], size)
            
        return filenames, captions, bboxes, filenames_wrong, bboxes_wrong
    
    def save(self, saver, checkpoint_dir, model_name, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step = step)
    
    def load(self, saver, checkpoint_dir):
        print(' [*] Reading checkpoints...')
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(' [*] Success to read {}'.format(ckpt_name))
            return True
        else:
            print(' [*] Failed to find a checkpoint')
            return False
