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

from coco_manager import COCOManager
from cub_manager import CUBManager
from optim_manager import OptimManager
from summary_manager import SummaryManager

class DeConvNET(object):
    
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

        logger.info('Loading Data...')
        if 'MSCOCO' in config.dataset:
            self.embedding_dim = 2400
            self.train_paths = ['captions_train2014_airplane.npy']
            self.val_paths = ['captions_val2014_airplane.npy']
            self.cocoManager = COCOManager()
            #self.get_data = self.cocoManager.get_data
        elif 'CUB' in config.dataset:
            if config.is_skip_thoughts:
                self.embedding_dim = 2400
                self.augment_dim = int(self.embedding_dim / 8)
                self.get_data_train = (CUBManager('/data1/BIRD')).get_data
                self.get_data_val = (CUBManager('/data1/BIRD', size = config.batch_size, train = False)).get_data
            else:
                self.embedding_dim = 1024
                self.augment_dim = int(self.embedding_dim / 8)
                self.get_data_train = (CUBManager('/data1/BIRD', is_skip_thoughts = False)).get_data
                self.get_data_val = (CUBManager('/data1/BIRD', is_skip_thoughts = False, train = False)).get_data
        logger.info('Loading Data...Done!')
            
        if 'mse' in config.criterionGAN:
            self.criterionGAN = mse_criterion
        elif 'sce' in config.criterionGAN:
            self.criterionGAN = sce_criterion
        
        self.fake_pool = fakePool(config.pool_max)

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
        self.bn_train_phase = tf.placeholder(tf.bool, name = 'phase')
        self.lr_g = tf.placeholder(tf.float32, name = 'learning_rate_g')
        self.lr_d = tf.placeholder(tf.float32, name = 'learning_rate_d')
        self.dr_rate = tf.placeholder(tf.float32, name = 'dropout_rate')
    
        ############# input #############
        self.images_real = tf.placeholder(tf.float32, [self.config.batch_size] + [self.config.image_size, self.config.image_size, self.config.c_dim], name = 'images_real')
        self.images_wrong = tf.placeholder(tf.float32, [self.config.batch_size] + [self.config.image_size, self.config.image_size, self.config.c_dim], name = 'images_wrong')
    
        self.captions = tf.placeholder(tf.float32, [self.config.batch_size] + [1, 1, self.embedding_dim], name = 'captions')
    
        self.noises = tf.placeholder(tf.float32, [self.config.batch_size] + [1, 1, self.config.noise_dim], name = 'noises')
        self.noises_zero = tf.zeros_like(self.noises)
    
        ########### GENERATOR ###########
        with tf.variable_scope('Generator'):
            logger.info('Initializing Generator...')
            with tf.variable_scope('Generator_AB') as scope:
                self.conditions_temp, self.kl_loss = self.augment_condition(self.captions, scope, reuse = False)
                self.conditions = tf.concat([self.conditions_temp, self.noises], axis = 3)
                self.images_fake = self.generator(self.conditions, bn_train_phase = self.bn_train_phase, scope = scope, reuse = False)
                self.images_fake_zero = self.generator(tf.concat([self.conditions_temp, self.noises_zero], axis = 3, name = 'condition'), bn_train_phase = False, scope = scope, reuse = True)
        
            with tf.variable_scope('Generator_BA') as scope:
                self.conditions_cycle, self.captions_cycle = self.generator_caption(self.images_fake, bn_train_phase = self.bn_train_phase, scope = scope, reuse = False)
                self.conditions_fake, self.captions_fake = self.generator_caption(self.images_real, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
        
            with tf.variable_scope('Generator_AB') as scope:
                self.images_cycle = self.generator(self.conditions_fake, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
            logger.info('Initializing Generator... DONE')
    
        ########## Discriminator ##########
        logger.info('Initializing Discriminator...')
        with tf.variable_scope('Discriminator'):
            with tf.variable_scope('Discriminator_R') as scope:
                self.images_fake_sample = tf.placeholder(tf.float32, [self.config.batch_size] + [self.config.image_size, self.config.image_size, self.config.c_dim], name = 'images_fake_sample')
                self.captions_fake_sample = tf.placeholder(tf.float32, [self.config.batch_size] + [1, 1, self.embedding_dim], name = 'captions_fake_sample')
            
                self.d_logits_relation_real_RR = self.discriminator_relation(self.images_real, self.captions, bn_train_phase = self.bn_train_phase, scope = scope, reuse = False)
                self.d_logits_relation_fake_WR = self.discriminator_relation(self.images_wrong, self.captions, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
                self.d_logits_relation_fake_FR = self.discriminator_relation(self.images_fake, self.captions, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
                self.d_logits_relation_fake_FR_sample = self.discriminator_relation(self.images_fake_sample, self.captions, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
                self.d_logits_relation_fake_RF = self.discriminator_relation(self.images_real, self.captions_fake, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
                self.d_logits_relation_fake_RF_sample = self.discriminator_relation(self.images_real, self.captions_fake_sample, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
        
            with tf.variable_scope('Discriminator_B') as scope:
                self.d_logits_content_real_B = self.discriminator_image(self.images_real, bn_train_phase = self.bn_train_phase, scope = scope, reuse = False)
                self.d_logits_content_fake_B = self.discriminator_image(self.images_fake, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
                self.d_logits_content_fake_B_sample = self.discriminator_image(self.images_fake_sample, bn_train_phase =  self.bn_train_phase, scope = scope, reuse = True)
        
            with tf.variable_scope('Discriminator_A') as scope:
                self.d_logits_content_real_A = self.discriminator_caption(self.captions, bn_train_phase = self.bn_train_phase, scope = scope, reuse = False)
                self.d_logits_content_fake_A = self.discriminator_caption(self.captions_fake, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
                self.d_logits_content_fake_A_sample = self.discriminator_caption(self.captions_fake_sample, bn_train_phase = self.bn_train_phase, scope = scope, reuse = True)
        logger.info('Initializing Discriminator... DONE')
        
        logger.info('Initializing NETWORK... DONE\n')
    
    def build_loss(self):
        logger.info('Initializing LOSS...')
        labels_real = tf.random_uniform([self.config.batch_size, 1], 0.7, 1.2)
        labels_fake = tf.random_uniform([self.config.batch_size, 1], 0.0, 0.3)
        #labels_real = tf.ones([self.config.batch_size, 1])
        #labels_fake = tf.zeros([self.config.batch_size, 1])

        # discriminator loss
        with tf.variable_scope('D_R'):
            self.d_loss_relation_real_RR = self.criterionGAN(logits = self.d_logits_relation_real_RR, labels = labels_real, name = self.summaryManager.get_sum_marked_name('7_d_RR_loss'))
            self.d_loss_relation_fake_WR = self.criterionGAN(logits = self.d_logits_relation_fake_WR, labels = labels_fake, name = self.summaryManager.get_sum_marked_name('6_d_WR_loss'))
            self.d_loss_relation_fake_FR = self.criterionGAN(logits = self.d_logits_relation_fake_FR_sample, labels = labels_fake, name = self.summaryManager.get_sum_marked_name('5_d_FR_loss'))
            self.d_loss_relation_fake_RF = self.criterionGAN(logits = self.d_logits_relation_fake_RF_sample, labels = labels_fake, name = self.summaryManager.get_sum_marked_name('4_d_RF_loss'))
        
            self.d_loss_relation_real = tf.identity(self.d_loss_relation_real_RR, name = self.summaryManager.get_sum_marked_name('3_d_total_real_loss'))
            self.d_loss_relation_fake = tf.identity((self.d_loss_relation_fake_WR + self.d_loss_relation_fake_FR + self.d_loss_relation_fake_RF) / 3., name = self.summaryManager.get_sum_marked_name('2_d_total_fake_loss'))
            self.d_loss_R = tf.identity(self.d_loss_relation_real + self.d_loss_relation_fake, name = self.summaryManager.get_sum_marked_name('1_d_total_loss_R'))
            
        with tf.variable_scope('D_B'):
            self.d_loss_content_real_B = self.criterionGAN(logits = self.d_logits_content_real_B, labels = labels_real, name = self.summaryManager.get_sum_marked_name('3_d_content_real_loss_B'))
            self.d_loss_content_fake_B = self.criterionGAN(logits = self.d_logits_content_fake_B_sample, labels = labels_fake, name = self.summaryManager.get_sum_marked_name('2_d_content_fake_loss_B'))
            self.d_loss_B = tf.add(self.d_loss_content_real_B, self.d_loss_content_fake_B, name = self.summaryManager.get_sum_marked_name('1_d_totla_loss_B'))
            
        with tf.variable_scope('D_A'):
            self.d_loss_content_real_A = self.criterionGAN(logits = self.d_logits_content_real_A, labels = labels_real, name = self.summaryManager.get_sum_marked_name('3_d_content_real_loss_A'))
            self.d_loss_content_fake_A = self.criterionGAN(logits = self.d_logits_content_fake_A_sample, labels = labels_fake, name = self.summaryManager.get_sum_marked_name('2_d_content_fake_loss_A'))
            self.d_loss_A = tf.add(self.d_loss_content_real_A, self.d_loss_content_fake_A, name = self.summaryManager.get_sum_marked_name('1_d_total_loss_A'))

        with tf.variable_scope('D_total'):
            self.d_loss = self.d_loss_R + self.d_loss_B + self.d_loss_A
    
        ############### generator loss ###############
        with tf.variable_scope('G_AB'):
            self.g_kl_loss = tf.identity(self.kl_loss, name = self.summaryManager.get_sum_marked_name('5_g_kl_loss'))
            self.g_loss_const_A = tf.identity(abs_criterion(self.conditions, self.conditions_cycle), name = self.summaryManager.get_sum_marked_name('4_g_cycle_loss_A'))
            self.g_loss_relation_gan_AB = self.criterionGAN(logits = self.d_logits_relation_fake_FR, labels = labels_real, name = self.summaryManager.get_sum_marked_name('3_g_gan_relation_loss_AB'))
            self.g_loss_content_gan_AB = self.criterionGAN(logits = self.d_logits_content_fake_B, labels = labels_real, name = self.summaryManager.get_sum_marked_name('2_g_gan_content_loss_AB'))
            self.g_loss_AB = tf.identity(self.g_kl_loss + self.g_loss_const_A + self.g_loss_relation_gan_AB + self.g_loss_content_gan_AB, name = self.summaryManager.get_sum_marked_name('1_g_total_loss_AB'))
    
        with tf.variable_scope('G_BA'):
            self.g_loss_const_B = tf.identity(abs_criterion(self.images_real, self.images_cycle), name = self.summaryManager.get_sum_marked_name('4_g_cycle_loss_B'))
            self.g_loss_relation_gan_BA = self.criterionGAN(logits = self.d_logits_relation_fake_RF, labels = labels_real, name = self.summaryManager.get_sum_marked_name('3_g_gan_relation_loss_BA'))
            self.g_loss_content_gan_BA = self.criterionGAN(logits = self.d_logits_content_fake_A, labels = labels_real, name = self.summaryManager.get_sum_marked_name('2_g_gan_content_loss_BA'))
            self.g_loss_BA = tf.identity(self.g_loss_const_B + self.g_loss_relation_gan_BA + self.g_loss_content_gan_BA, name = self.summaryManager.get_sum_marked_name('1_g_total_loss_BA'))
    
        with tf.variable_scope('G_total'):
            self.g_loss = self.g_loss_AB + self.g_loss_BA
        logger.info('Initializing LOSS... DONE\n')
        
    def build_variables(self):
        logger.info('Initializing NETWORK VARIABLE...')
        self.t_vars = tf.trainable_variables()
        self.g_vars = [var for var in self.t_vars if 'Generator' in var.name]
        self.d_vars = [var for var in self.t_vars if 'Discriminator' in var.name]
    
        self.g_vars_AB = [var for var in self.t_vars if 'Generator_AB' in var.name]
        self.g_vars_BA = [var for var in self.t_vars if 'Generator_BA' in var.name]
        self.d_vars_R = [var for var in self.t_vars if 'Discriminator_R' in var.name]
        self.d_vars_A = [var for var in self.t_vars if 'Discriminator_A' in var.name]
        self.d_vars_B = [var for var in self.t_vars if 'Discriminator_B' in var.name]
        logger.info('Initializing NETWORK VARIABLE... DONE\n')
        
        show_all_variables(self.g_vars, 'Generator')
        show_all_variables(self.d_vars, 'Disciriminator')
        show_all_variables(verbose = False)

        logger.info('Initializing SUMMARY VARIABLE...')
        self.build_sum_var()
        logger.info('Initializing SUMMARY VARIABLE... DONE\n')
        
    def build_sum_var(self):
        logger.info('Initializing SUMMARY VARIABLE...')
        with tf.variable_scope('fake'):
            self.summaryManager.add_image_sum(self.images_fake, 'fake')
        with tf.variable_scope('fake_zero'):
            self.summaryManager.add_image_sum(self.images_fake_zero, 'fake_zero')
        with tf.variable_scope('real'):
            self.summaryManager.add_image_sum(self.images_real, 'real')
            self.summaryManager.add_histogram_sum(self.captions, 'caption_real')
            self.summaryManager.add_histogram_sum(self.captions_fake, 'caption_fake')
            self.summaryManager.add_histogram_sum(self.conditions, 'condition_real')
            self.summaryManager.add_histogram_sum(self.conditions_fake, 'condition_fake')
    
        self.summaryManager.set_sum_vars(self.sess.graph)
        logger.info('Initializing SUMMARY VARIABLE... DONE')
        
    def build_optim(self):
        ###### Optimizer for network ######
        logger.info('Initializing Optimizer ...')
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            with tf.variable_scope('Optimizer'):
                self.d_optim_R, self.grad_and_vars_d_R = self.optimManager.get_optim_and_grad_vars(self.lr_d, self.d_loss_R, self.d_vars_R)
                self.d_optim_A, self.grad_and_vars_d_A = self.optimManager.get_optim_and_grad_vars(self.lr_d, self.d_loss_A, self.d_vars_A)
                self.d_optim_B, self.grad_and_vars_d_B = self.optimManager.get_optim_and_grad_vars(self.lr_d, self.d_loss_B, self.d_vars_B)
                self.g_optim_AB, self.grad_and_vars_g_AB = self.optimManager.get_optim_and_grad_vars(self.lr_g, self.g_loss_AB, self.g_vars_AB)
                self.g_optim_BA, self.grad_and_vars_g_BA = self.optimManager.get_optim_and_grad_vars(self.lr_g, self.g_loss_BA, self.g_vars_BA)
        logger.info('Initializing Optimizer ... DONE\n')
    
    def build_summary(self):
        logger.info('Initializing Summary ...')
        self.itm_sum = self.summaryManager.get_merged_summary() # images, histograms
        self.g_loss_sum_AB = self.summaryManager.get_merged_summary(self.g_vars_AB, grad_norm = get_grads_norm(self.grad_and_vars_g_AB), name = 'G_AB')
        self.g_loss_sum_BA = self.summaryManager.get_merged_summary(self.g_vars_BA, grad_norm = get_grads_norm(self.grad_and_vars_g_BA), name = 'G_BA')
        self.d_loss_sum_R = self.summaryManager.get_merged_summary(self.d_vars_R, grad_norm = get_grads_norm(self.grad_and_vars_d_R), name = 'D_R')
        self.d_loss_sum_A = self.summaryManager.get_merged_summary(self.d_vars_A, grad_norm = get_grads_norm(self.grad_and_vars_d_A), name = 'D_A')
        self.d_loss_sum_B = self.summaryManager.get_merged_summary(self.d_vars_B, grad_norm = get_grads_norm(self.grad_and_vars_d_B), name = 'D_B')
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
        lr_d = self.config.lr_d
        lr_g = self.config.lr_g
        
        # load data
        val_files, val_captions, val_bboxes, val_files_wrong, val_bboxes_wrong = self.get_data_val()
        
        logger.info('Training Starts!')
        it_epoch = trange(self.config.epoch, ncols = 100, initial = 0, desc = 'Epoch')
        for epoch in it_epoch:
            
            train_files, train_captions, train_bboxes, train_files_wrong, train_bboxes_wrong = self.get_data_train()

            batch_idxs = min(len(train_files), self.config.train_size) // self.config.batch_size
            it_train = trange(batch_idxs, ncols = 100, initial = 0, desc = '[Train]')
            for idx in it_train:
                
                batch_range = np.arange(idx * self.config.batch_size, (idx + 1) * self.config.batch_size)

                batch_images = get_images(train_files, batch_range, self.config.is_crop, self.config.crop_mode, train_bboxes, self.config.image_size, None, self.config.is_grayscale)
                batch_images_wrong = get_images(train_files_wrong, batch_range, self.config.is_crop, self.config.crop_mode, train_bboxes_wrong, self.config.image_size, None, self.config.is_grayscale)
                
                captions = np.take(train_captions, batch_range, axis = 0).reshape((self.config.batch_size, 1, 1, self.embedding_dim))
                noises = np.random.normal(loc = 0, scale = 1, size=(self.config.batch_size, 1, 1, self.config.noise_dim))

                lr_d = get_learning_rate(lr_d, epoch, counter, self.config.lr_decay_steps, self.config.lr_decay_rate)
                lr_g = get_learning_rate(lr_g, epoch, counter, self.config.lr_decay_steps, self.config.lr_decay_rate)

                # Forward Generator AB
                fake_A, fake_B = self.sess.run([self.captions_fake, self.images_fake],
                                               feed_dict = {self.dr_rate: self.config.dr_rate,
                                                            self.images_real: batch_images,
                                                            self.captions: captions, self.noises: noises,
                                                            self.bn_train_phase: False})
                fake_A, fake_B = self.fake_pool(fake_A, fake_B)

                # Update Discriminator R
                for _ in np.arange(self.config.num_D_train):
                    _, d_loss_R, summary_str = self.sess.run([self.d_optim_R, self.d_loss_R, self.d_loss_sum_R],
                                                             feed_dict = {
                                                                 self.lr_d: lr_d, self.dr_rate: self.config.dr_rate,
                                                                 self.images_real: batch_images,
                                                                 self.images_wrong: batch_images_wrong,
                                                                 self.images_fake_sample: fake_B,
                                                                 self.captions: captions, self.noises: noises,
                                                                 self.captions_fake_sample: fake_A,
                                                                 self.bn_train_phase: True})
                self.writer.add_summary(summary_str, counter)
                
                # Update Discriminator A
                for _ in np.arange(self.config.num_D_train):
                    _, d_loss_A, summary_str = self.sess.run([self.d_optim_A, self.d_loss_A, self.d_loss_sum_A],
                                                             feed_dict = {
                                                                 self.lr_d: lr_d, self.dr_rate: self.config.dr_rate,
                                                                 self.captions: captions, self.noises: noises,
                                                                 self.captions_fake_sample: fake_A,
                                                                 self.bn_train_phase: True})
                self.writer.add_summary(summary_str, counter)
                
                # Update Discriminator B
                for _ in np.arange(self.config.num_D_train):
                    _, d_loss_B, summary_str = self.sess.run([self.d_optim_B, self.d_loss_B, self.d_loss_sum_B],
                                                             feed_dict = {
                                                                 self.lr_d: lr_d, self.dr_rate: self.config.dr_rate,
                                                                 self.images_real: batch_images,
                                                                 self.images_wrong: batch_images_wrong,
                                                                 self.images_fake_sample: fake_B,
                                                                 self.bn_train_phase: True})
                self.writer.add_summary(summary_str, counter)
                
                for _ in np.arange(self.config.num_G_train):
                    # Update Generator AB
                    _, g_loss_AB, summary_str_AB = self.sess.run([self.g_optim_AB, self.g_loss_AB, self.g_loss_sum_AB],
                                                              feed_dict = {
                                                                  self.lr_g: lr_g, self.dr_rate: self.config.dr_rate,
                                                                  self.images_real: batch_images,
                                                                  self.captions: captions, self.noises: noises,
                                                                  self.bn_train_phase: True}) # batch moving avg already calculated at step 1
                
                    # Update Generator BA
                    _, g_loss_BA, summary_str_BA = self.sess.run([self.g_optim_BA, self.g_loss_BA, self.g_loss_sum_BA],
                                                              feed_dict = {
                                                                  self.lr_g: lr_g, self.dr_rate: self.config.dr_rate,
                                                                  self.images_real: batch_images,
                                                                  self.captions: captions, self.noises: noises,
                                                                  self.bn_train_phase: True}) # batch moving avg already calculated at step 1
                    
                self.writer.add_summary(summary_str_AB, counter)
                self.writer.add_summary(summary_str_BA, counter)

                # write img summary
                summary_str = self.sess.run(self.itm_sum,
                                            feed_dict = {
                                                self.dr_rate: self.config.dr_rate,
                                                self.images_real: batch_images,
                                                self.captions: captions, self.noises: noises,
                                                self.bn_train_phase: False})
                self.writer.add_summary(summary_str, counter)

                d_loss = d_loss_R + d_loss_B + d_loss_A
                g_loss = g_loss_AB + g_loss_BA
                it_train.set_description(('[Train] epoch: %d, g_loss: %.4f, d_loss: %.4f' % (epoch, g_loss, d_loss)))
                
                counter += 1
            
            # validation
            if np.mod(epoch, 1) == 0:
                val_batch_idxs = 0
                val_batch_range = np.arange(val_batch_idxs * self.config.batch_size, (val_batch_idxs+ 1) * self.config.batch_size)
    
                val_images = get_images(val_files, val_batch_range, self.config.is_crop, self.config.crop_mode, val_bboxes, self.config.image_size, None, self.config.is_grayscale)
                val_images_wrong = get_images(val_files_wrong, val_batch_range, self.config.is_crop, self.config.crop_mode, val_bboxes_wrong, self.config.image_size, None, self.config.is_grayscale)
    
                val_captions = np.take(val_captions, val_batch_range, axis = 0).reshape((self.config.batch_size, 1, 1, self.embedding_dim))
                val_noises = np.random.normal(loc = 0, scale = 1, size=(self.config.batch_size, 1, 1, self.config.noise_dim))

                #res = self.get_gradient_penalty(self.config.dr_rate, val_images, val_captions, False)

                fake_A, fake_B = self.sess.run([self.captions_fake, self.images_fake],
                                       feed_dict = {self.dr_rate: self.config.dr_rate,
                                                    self.images_real: val_images,
                                                    self.captions: val_captions, self.noises: val_noises,
                                                    self.bn_train_phase: False})

                sample_normal, sample_zero, g_loss, d_loss = self.sess.run(
                    [self.images_fake, self.images_fake_zero, self.g_loss, self.d_loss],
                    feed_dict = {self.dr_rate: self.config.dr_rate,
                                 self.images_real: val_images, self.captions: val_captions,
                                 self.images_wrong: val_images_wrong, self.noises: val_noises,
                                 self.images_fake_sample: fake_B,
                                 self.captions_fake_sample: fake_A,
                                 self.bn_train_phase: False}
                )
                
                output_size = np.int32(np.ceil(np.sqrt(self.config.batch_size)))
    
                time_str = get_time_str()
                
                save_images(sample_normal, [output_size, output_size],
                            '{}/val_caption_normal_{:02d}_{}.png'.format(self.config.sample_dir, epoch, time_str))
                save_images(sample_zero, [output_size, output_size],
                            '{}/val_caption_zero_{:02d}_{}.png'.format(self.config.sample_dir, epoch, time_str))
                save_images(val_images, [output_size, output_size],
                            '{}/val_gt_{:02d}_{}.png'.format(self.config.sample_dir, epoch, time_str))
    
                it_epoch.set_description(('[Sample] epoch: %d, g_loss: %.4f, d_loss: %.4f' % (epoch, g_loss, d_loss)))
        
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
        
    def augment_condition(self, captions, scope = 'Condition_Augmentor', reuse = False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
                
            conditions = lrelu(conv2d(captions, self.augment_dim * 2, k_h = 1, k_w = 1, padding = 'VALID', name = 'ca_h0_linear_0'))
            mean, log_sigma = tf.split(conditions, 2, axis = 3)
            
            if self.config.is_CA:
                epsilon = tf.truncated_normal(tf.shape(mean))
                stddev = tf.exp(log_sigma)
                conditions = mean + stddev * epsilon
                kl_loss = KL_loss(mean, log_sigma)
            else:
                conditions = mean
                kl_loss = 0
            
            return conditions, self.config.KL_coeff * kl_loss
                
    def generator(self, conditions, bn_train_phase, scope = 'Generator', reuse = False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
            
            s = self.config.image_size #[224]
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) #[112, 56, 28, 14]

            h0 = conv2d(conditions, s16 * s16 * self.config.gf_dim * 8, k_h = 1, k_w = 1, padding = 'VALID', name = 'h0_linear_0')
            h0 = tf.reshape(h0, [self.config.batch_size, s16, s16, self.config.gf_dim * 8])
            h0 = tf.nn.relu(bnorm(h0, bn_train_phase, name = 'h0_bn_1'))
            
            h1 = deconv2d(h0, [self.config.batch_size, s8, s8, self.config.gf_dim * 4], name = 'h1_deconv_0')
            h1 = tf.nn.relu(tf.nn.dropout(bnorm(h1, bn_train_phase, name = 'h1_bn_1'), self.dr_rate))
            h1 = conv2d(h1, self.config.gf_dim, name = 'h1_conv_2')
            h1 = tf.nn.relu(tf.nn.dropout(bnorm(h1, bn_train_phase, name = 'h1_bn_3'), self.dr_rate))
            h1 = conv2d(h1, self.config.gf_dim, name = 'h1_conv_4')
            h1 = tf.nn.relu(bnorm(h1, bn_train_phase, name = 'h1_bn_5'))
            
            h2 = deconv2d(h1, [self.config.batch_size, s4, s4, self.config.gf_dim * 2], name = 'h2_deconv_0')
            h2 = tf.nn.relu(bnorm(h2, bn_train_phase, name = 'h2_bn_1'))
            h2 = conv2d(h2, self.config.gf_dim * 2, name = 'h2_conv_2')
            h2 = tf.nn.relu(bnorm(h2, bn_train_phase, name = 'h2_bn_3'))
            h2 = conv2d(h2, self.config.gf_dim * 2, name = 'h2_conv_4')
            h2 = tf.nn.relu(bnorm(h2, bn_train_phase, name = 'h2_bn_5'))
            
            h3 = deconv2d(h2, [self.config.batch_size, s2, s2, self.config.gf_dim], name = 'h3_deconv_0')
            h3 = tf.nn.relu(bnorm(h3, bn_train_phase, name = 'h3_bn_1'))
            h3 = conv2d(h3, self.config.gf_dim, name = 'h3_conv_2')
            h3 = tf.nn.relu(bnorm(h3, bn_train_phase, name = 'h3_bn_3'))
            h3 = conv2d(h3, self.config.gf_dim, name = 'h3_conv_4')
            h3 = tf.nn.relu(bnorm(h3, bn_train_phase, name = 'h3_bn_5'))
            
            h4 = deconv2d(h3, [self.config.batch_size, s, s, 3], name = 'h4_deconv_0')
            h4 = tf.nn.relu(bnorm(h4, bn_train_phase, name = 'h4_bn_1'))
            h4 = conv2d(h4, 3, name = 'h4_conv_2')
            h4 = tf.nn.relu(bnorm(h4, bn_train_phase, name = 'h4_bn_3'))
            h4 = conv2d(h4, 3, name = 'h4_conv_4')
            
            return tf.tanh(h4)
        
    def generator_caption(self, input, bn_train_phase, scope = 'Generator_Caption', reuse = False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
        
            h0 = conv2d(input, self.config.df_dim, name = 'h0_conv')
            h0 = tf.nn.relu(bnorm(h0, bn_train_phase, name = 'h0_bn'))
            h0 = tf.nn.avg_pool(h0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h0_pool') #112
        
            h1 = conv2d(h0, self.config.df_dim * 2, name = 'h1_conv')
            h1 = tf.nn.relu(tf.nn.dropout(bnorm(h1, bn_train_phase, name = 'h1_bn'), self.dr_rate))
            h1 = tf.nn.avg_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h1_pool') #56
        
            h2 = conv2d(h1, self.config.df_dim * 4, name = 'h2_conv')
            h2 = tf.nn.relu(tf.nn.dropout(bnorm(h2, bn_train_phase, name = 'h2_bn'), self.dr_rate))
            h2 = tf.nn.avg_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h2_pool') #28
        
            h3 = conv2d(h2, self.config.df_dim * 8, name = 'h3_conv')
            h3 = tf.nn.relu(bnorm(h3, bn_train_phase, name = 'h3_bn'))
            h3 = tf.nn.avg_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h3_pool') #14
        
            h4 = conv2d(h3, self.config.df_dim * 8, name = 'h4conv')
            h4 = tf.nn.relu(bnorm(h4, bn_train_phase, name = 'h4_bn'))
            image_embeddings = tf.nn.avg_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h4_pool') #7
            
            image_embeddings_h, image_embeddings_w = image_embeddings.get_shape().as_list()[1:3]
            
            h4 = tf.nn.relu(conv2d(image_embeddings, self.config.dfc_dim, k_h = image_embeddings_h, k_w = image_embeddings_w, padding = 'VALID', name = 'h4_linear_0'))
            h4 = tf.nn.relu(conv2d(h4, self.config.dfc_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h4_linear_1'))
            
            #captions_inverse_w_noise = tf.tanh(conv2d(h4, self.embedding_dim + self.config.noise_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h4_linear_2'))
            #captions_inverse, noise_inverse = tf.split(captions_inverse_w_noise, [self.embedding_dim, self.config.noise_dim], 3)
            #conditions_inverse, kl_loss_BA = self.augment_condition(captions_inverse_w_noise, scope = scope, reuse = reuse)
            #conditions_inverse_w_noise = tf.concat([conditions_inverse, noise_inverse], axis = 3)
            
            conditions_inverse_w_noise = conv2d(h4, self.augment_dim + self.config.noise_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h4_linear_2')
            conditions_inverse, noise_inverse = tf.split(conditions_inverse_w_noise, [self.augment_dim, self.config.noise_dim], 3)
            captions_inverse = conv2d(conditions_inverse, self.embedding_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h4_linear_3')
            
            return conditions_inverse_w_noise, captions_inverse

    def discriminator_relation(self, input, captions, bn_train_phase, scope = 'Discriminator_Relation', reuse = False):
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
        
            # reduce caption dim
            h0_0 = lrelu(conv2d(captions, self.augment_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h0_0_linear_0'))
            h0_0 = lrelu(conv2d(h0_0, self.augment_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h0_0_linear_1'))
            h0_0 = tf.reshape(h0_0, [self.config.batch_size, 1, 1, self.augment_dim])
        
            # concats embeddings
            image_embeddings_h, image_embeddings_w = image_embeddings.get_shape().as_list()[1:3]
            captions_replicated = tf.tile(h0_0, [1, image_embeddings_h, image_embeddings_w, 1])
            concated_activations = tf.concat([image_embeddings, captions_replicated], 3)
        
            h5 = conv2d(concated_activations, self.config.df_dim * 8, k_h = 1, k_w = 1, name = 'h5_conv_0')
            concated_activations = lrelu(bnorm(h5 , bn_train_phase, name = 'h5_bn_1'))
        
            h5 = lrelu(conv2d(concated_activations, self.config.dfc_dim, k_h = image_embeddings_h, k_w = image_embeddings_w, padding = 'VALID', name = 'h5_linear_2'))
            h5 = lrelu(conv2d(h5, self.config.dfc_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h5_linear_3'))
            h5 = conv2d(h5, 1, k_h = 1, k_w = 1, padding = 'VALID', name = 'd_h5_linear_4')
            logits_relation = tf.reshape(h5, [self.config.batch_size, -1])
        
            return logits_relation
        
    def discriminator_image(self, input, bn_train_phase, scope = 'Discriminator_Image', reuse = False):
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
            logits = tf.reshape(h4, [self.config.batch_size, -1])
        
            return logits
        
    def discriminator_caption(self, input, bn_train_phase, scope = 'Discriminator_captions', reuse = False):
        with tf.variable_scope(scope) as scope:
            if reuse:
                scope.reuse_variables()
        
            h0 = conv2d(input, self.config.dfc_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h0_linear_0')
            h0 = lrelu(bnorm(h0, bn_train_phase, name = 'h0_bn_1'))
            h0 = conv2d(h0, self.config.dfc_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h0_linear_2')
            h0 = lrelu(bnorm(h0, bn_train_phase, name = 'h0_bn_3'))
            h0 = conv2d(h0, self.config.dfc_dim, k_h = 1, k_w = 1, padding = 'VALID', name = 'h0_linear_4')
            h0 = lrelu(bnorm(h0, bn_train_phase, name = 'h0_bn_5'))
            
            h1 = conv2d(h0, self.config.dfc_dim/2, k_h = 1, k_w = 1, padding = 'VALID', name = 'h1_linear_0')
            h1 = lrelu(bnorm(h1, bn_train_phase, name = 'h1_bn_1'))
            h1 = conv2d(h1, self.config.dfc_dim/2, k_h = 1, k_w = 1, padding = 'VALID', name = 'h1_linear_2')
            h1 = lrelu(bnorm(h1, bn_train_phase, name = 'h1_bn_3'))
            
            h2 = conv2d(h1, self.config.dfc_dim/4, k_h = 1, k_w = 1, padding = 'VALID', name = 'h2_linear_0')
            h2 = lrelu(bnorm(h2, bn_train_phase, name = 'h2_bn_1'))
            h2 = conv2d(h2, self.config.dfc_dim/4, k_h = 1, k_w = 1, padding = 'VALID', name = 'h2_linear_2')
            h2 = lrelu(bnorm(h2, bn_train_phase, name = 'h2_bn_3'))
            
            h3 = conv2d(h2, self.config.dfc_dim/8, k_h = 1, k_w = 1, padding = 'VALID', name = 'h3_linear_0')
            #h3 = lrelu(bnorm(h3, bn_train_phase, name = 'h3_bn_1'))
            h3 = lrelu(h3, name = 'h3_lrelu_1')
            h3 = conv2d(h3, self.config.dfc_dim/8, k_h = 1, k_w = 1, padding = 'VALID', name = 'h3_linear_2')
            #h3 = lrelu(bnorm(h3, bn_train_phase, name = 'h3_bn_3'))
            h3 = lrelu(h3, name = 'h3_lrelu_3')

            h3 = conv2d(h3, 1, k_h = 1, k_w = 1, padding = 'VALID', name = 'h3_linear_4')
            
            caption_logits = tf.reshape(h3, [self.config.batch_size, -1])
        
            return caption_logits
    
    def get_data(self, data_paths, size = None):
        if self.config.dataset == 'MSCOCO':
            '''
            # Load pre-computed captions
            #filenames, captions, bboxes, filenames_wrong, bboxes_wrong = self.cocoManager.load_captions(data_paths[0])
        
            # compute and Load captions
            categories = ['airplane']
            train_data = self.cocoManager.get_captions('/data1/mscoco', 'train2014', categories)
            '''
        
        elif self.config.dataset == 'CUB':
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
        
    '''
    def get_gradient_penalty(self, dr_rate, image_real, captions, bn_train_phase):
        pr = self.sess.partial_run_setup([self.gradient_penalty],
                                         [self.dr_rate,
                                          self.images_real,
                                          self.captions,
                                          self.bn_train_phase])

        return self.sess.partial_run(pr, self.gradient_penalty,
                                     feed_dict = {self.dr_rate: dr_rate,
                                                  self.images_real: image_real,
                                                  self.captions: captions,
                                                  self.bn_train_phase: train_phase})
    '''
