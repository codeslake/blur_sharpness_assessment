from __future__ import division
import os
import time
import datetime
from glob import glob
from six.moves import xrange
import fnmatch
import pickle

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
        self.get_data_train = (AVAManager('/data1/AVA', config.type, train = True)).get_data
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
        if self.config.is_train:
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
            self.logits_pred, self.scores_pred = self.regressor(self.images, bn_train_phase = self.bn_train_phase, scope = scope, reuse = False)
        logger.info('Initializing Discriminator... DONE')
        
        logger.info('Initializing NETWORK... DONE\n')
    
    def build_loss(self):
        logger.info('Initializing LOSS...')
        # regressor loss
        with tf.variable_scope('R_loss'):
            self.r_loss = self.criterion(logits = self.logits_pred, labels = self.scores_expt, name = self.summaryManager.get_sum_marked_name('1_r_loss'))
            #self.r_loss = self.criterion(logits = self.scores_pred, labels = self.scores_expt, name = self.summaryManager.get_sum_marked_name('1_r_loss'))
    
        logger.info('Initializing LOSS... DONE\n')
        
    def build_variables(self):
        logger.info('Initializing NETWORK VARIABLE...')
        self.t_vars = tf.trainable_variables()
        bn_vars = [var for var in tf.contrib.graph_editor.get_tensors(self.sess.graph) if 'moving_mean' in var.name]
        print "****************"
        print type(self.t_vars)
        print "****************"
        self.t_vars = self.t_vars + bn_vars
        
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
            labels = tf.constant([['score_expt'],['score_pred'],['logits_pred']], tf.string)
            score_expt = tf.transpose(self.scores_expt, perm = [1, 0])
            score_pred = tf.transpose(self.scores_pred, perm = [1, 0])
            logits_pred = tf.transpose(self.logits_pred, perm = [1, 0])
            #concatenated = tf.concat(labels, tf.concat(score_expt, tf.concat(score_pred, logits_pred, 0), 0), 1)
            concatenated = tf.concat([labels, tf.as_string(tf.concat([score_expt, tf.concat([score_pred, logits_pred], 0)], 0), precision = -3)], 1)
            
            self.summaryManager.add_text_sum(concatenated, 'outputs')
            #self.summaryManager.add_text_sum(tf.as_string(tf.transpose(self.scores_expt, perm = [1, 0]), precision = -3), 'score_expt')
            #self.summaryManager.add_text_sum(tf.as_string(tf.transpose(self.scores_pred, perm = [1, 0]), precision = -3), 'score_pred')
            #self.summaryManager.add_text_sum(tf.as_string(tf.transpose(self.logits_pred, perm = [1, 0]), precision = -3), 'logits_pred')
    
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
        self.r_loss_sum = self.summaryManager.get_merged_summary(self.r_vars, grad_norm = get_grads_norm(self.grad_and_vars_r), name = 'R_loss')
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
            self.save(self.saver_train, self.config.checkpoint_dir, self.config.model_name, counter)
            
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
            return

        with open('/data1/AVA/AVA_classified/255000/filenames_test.pickle', 'rb') as f:
            filenames = np.array(pickle.load(f)) #[file_num]

        with open('/data1/AVA/AVA_classified/255000/scores_test.pickle', 'rb') as f:
            scores = np.array(pickle.load(f)) #[file_num]

        file_num = filenames.shape[0]
        print "[shape]: " + str(filenames.shape[0])
        images = get_images(filenames, np.arange(file_num), self.config.is_crop, self.config.crop_mode, self.config.image_size, None, self.config.is_grayscale)
        
        for i in np.arange(file_num):
            
            score = self.sess.run(self.logits_pred,
                                  feed_dict = {self.dr_rate: self.config.dr_rate, self.bn_train_phase: False,
                                               self.images: np.expand_dims(images[i], axis = 0)})

            print '[filename]: ' + filenames[i] + ' [score_predicted]: ' + str(score) + " [ground_truth]: " + str(scores[i])
            
    def regressor(self, input, bn_train_phase, scope = 'Regressor', reuse = False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
        
            input = conv2d(input, self.config.df_dim, name = 'h0_input_0')
            
            h0 = conv2d(input, self.config.df_dim, name = 'h0_conv_0')
            h0 = bnorm(h0, bn_train_phase, name = 'h0_bn_1')
            h0 = lrelu(h0, name = 'h0_relu_2')
            h0 = conv2d(h0, self.config.df_dim, name = 'h0_conv_3')
            h0 = bnorm(h0, bn_train_phase, name = 'h0_bn_4')
            h0 = tf.add(h0, input, name = 'h0_add_5')
            h0 = tf.nn.max_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h0_pool_6') # 512 X 512 X 64
            
            h0 = conv2d(h0, self.config.df_dim * 2, name = 'h1_input_0')
            
            h1 = conv2d(h0, self.config.df_dim * 2, name = 'h1_conv_0')
            h1 = bnorm(h1, bn_train_phase, name = 'h1_bn_1')
            h1 = lrelu(h1, name = 'h1_relu_2')
            h1 = conv2d(h1, self.config.df_dim * 2, name = 'h1_conv_3')
            h1 = bnorm(h1, bn_train_phase, name = 'h1_bn_4')
            h1 = tf.add(h1, h0, name = 'h1_add_5')
            h1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h1_pool_6') # 256 X 256 X 128

            h1 = conv2d(h1, self.config.df_dim * 4, name = 'h2_input_0')
            
            h2 = conv2d(h1, self.config.df_dim * 4, name = 'h2_conv_0')
            h2 = bnorm(h2, bn_train_phase, name = 'h2_bn_1')
            h2 = lrelu(h2, name = 'h2_relu_2')
            h2 = conv2d(h2, self.config.df_dim * 4, name = 'h2_conv_3')
            h2 = bnorm(h2, bn_train_phase, name = 'h2_bn_4')
            h2 = tf.add(h2, h1, name = 'h2_add_5')
            h2 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h0_pool_6') # 128 X 128 X 256

            h2 = conv2d(h2, self.config.df_dim * 8, name = 'h3_input_0')
            
            h3 = conv2d(h2, self.config.df_dim * 8, name = 'h3_conv_0')
            h3 = bnorm(h3, bn_train_phase, name = 'h3_bn_1')
            h3 = lrelu(h3, name = 'h3_relu_2')
            h3 = conv2d(h3, self.config.df_dim * 8, name = 'h3_conv_3')
            h3 = bnorm(h3, bn_train_phase, name = 'h3_bn_4')
            h3 = tf.add(h3, h2, name = 'h3_add_5')
            h3 = tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h3_pool_6') # 64 X 64 X 512
        
            h, w = h3.get_shape().as_list()[1:3]
            h4 = conv2d(h3, self.config.dfc_dim, k_h = h, k_w = w, padding = 'VALID', name = 'h4_linear_0') # 1 X 1 X 1024
            h4 = lrelu(h4, name = 'h4_relu_1')
            h4 = conv2d(h4, self.config.dfc_dim / 2, k_h = 1, k_w = 1, padding = 'VALID', name = 'h4_linear_2') # 1 X 1 X 512
            h4 = lrelu(h4, name = 'h4_relu_3')
            h4 = conv2d(h4, self.config.dfc_dim / 4, k_h = 1, k_w = 1, padding = 'VALID', name = 'h4_linear_4') # 1 X 1 X 256
            h4 = lrelu(h4, name = 'h4_relu_5')
            h4 = conv2d(h4, 1, k_h = 1, k_w = 1, padding = 'VALID', name = 'h4_linear_6') # 1 X 1
            logits = tf.reshape(h4, [self.config.batch_size, -1])
            scores = tf.tanh(logits)
        
            return logits, scores
        
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
