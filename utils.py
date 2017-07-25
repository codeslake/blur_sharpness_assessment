from __future__ import division
import scipy.misc
import numpy as np
import tensorflow as tf
import datetime
import os
import fnmatch
import pprint
import copy

pp = pprint.PrettyPrinter()

def get_images(image_paths, idx_range, is_crop = True, crop_mode = 'random', resize_h = 224, resize_w = None, is_grayscale = False):
    resize_w = resize_h if resize_w is None else resize_w
    
    image_paths = image_paths.flatten()
    image_paths_batch = np.take(image_paths, idx_range, axis = 0)
    
    images = []
    for i in np.arange(len(image_paths_batch)):
        image_path = image_paths_batch[i]
        #print '[file name]', image_path
        if not os.path.isfile(image_path):
            continue
            
        image = imread(image_path, is_grayscale)
        if image is None:
            print image_path
            continue
            
        images.extend([augment(image, is_crop, crop_mode, resize_h, resize_w)])
        
    if is_grayscale:
        images = np.array(images).astype(np.float32)[:, :, :, None]
    else:
        images = np.array(images).astype(np.float32)

    return images
    
def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imread(path, is_grayscale = False):
    if is_grayscale:
        try:
            return scipy.misc.imread(path, flatten = True).astype(np.float)
        except:
            return None
    else:
        try:
            return scipy.misc.imread(path, mode = 'RGB').astype(np.float)
        except:
            return None

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w, resize_h, resize_w):
    start_idx = x.shape[:2] - np.array([crop_h, crop_w])
    end_idx = ((start_idx - np.array([crop_h, crop_w]))/2.).astype(int)
    
    return scipy.misc.imresize(x[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]], [resize_h, resize_w])

def random_crop(x, crop_h, crop_w, resize_h, resize_w):
    rand = np.random.normal(loc=0.5, scale=0.5)
    rand = 0.0 if rand < 0 else 1.0 if rand > 1.0 else rand
    start_idx = np.round((x.shape[:2] - np.array([crop_h, crop_w])) * rand).astype(int)
    end_idx = (start_idx + np.array([crop_h, crop_w])).astype(int)

    rand = np.random.normal(loc=0.5, scale=0.5)
    rand = 0.0 if rand < 0 else 1.0 if rand > 1.0 else rand
    x = np.flip(x, 1) if rand > 0.5 else x
    
    return scipy.misc.imresize(x[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]], [resize_h, resize_w])
        
def augment(image, is_crop, crop_mode, resize_h, resize_w):
    # npx : # of pixels width/height of image
    h, w = image.shape[:2]
    npx = w if h > w else h
        
    if is_crop:
        if crop_mode is 'random':
            image = random_crop(image, npx, npx, resize_h, resize_w)
        elif crop_mode is 'center':
            image = center_crop(image, npx, npx, resize_h, resize_w)
    else:
        image = scipy.misc.imresize(image, [resize_h, resize_w])

    # normalize between range
    '''
    range = [-1, 1]
    min = cropped_image.min()
    max = cropped_image.max()
    image = (range[1] - range[0]) * (image - min)/(max - min) + range[0]
    '''
    image = (image * 2.) / 255. - 1
    
    return image

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def show_all_variables(var = None, name = None, verbose = True):
    if var is None:
        var = tf.trainable_variables()
    if name is not None:
        print "*******************"
        print name
        print "*******************"
    else:
        name = 'Total'
    total_count = 0
    for idx, op in enumerate(var):
        shape = op.get_shape()
        count = np.prod(shape)
        if verbose:
            print "[%2d] %s %s = %0.2f MB" % (idx, op.name, shape, (int(count) * 8) / float(1024**2))
        total_count += int(count)
    print "[%s] variable size: %s MB" % (name, "{:,}".format((total_count * 8) / np.float32(1024**2)))
    
def get_image_names_in_dir(path):
    file_names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.JPEG'):
            file_names.append(filename)
        for filename in fnmatch.filter(filenames, '*.PNG'):
            file_names.append(filename)
        for filename in fnmatch.filter(filenames, '*.jpeg'):
            file_names.append(filename)
        for filename in fnmatch.filter(filenames, '*.png'):
            file_names.append(filename)
        for filename in fnmatch.filter(filenames, '*.jpg'):
            file_names.append(filename)
    
    return file_names

def get_image_path_in_dir(path):
    file_path = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '/*.JPEG'):
            file_path.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '/*.PNG'):
            file_path.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '/*.jpeg'):
            file_path.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '/*.png'):
            file_path.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '*.jpg'):
            file_path.append(os.path.join(root, filename))
    
    return file_path

def get_grads_norm(grad_and_vars, name = 'norm'):
    grads = grad_and_vars[:][0]
    return tf.norm(grads, name=name)

def get_learning_rate(lr, epoch, counter, decay_steps, decay_rate):
    learning_rate = lr
    
    if (epoch + 1) % decay_steps == 0:
        learning_rate = learning_rate * decay_rate
    
    return learning_rate

def get_time_str(format='%g%m%d_%H%M'):
    return datetime.datetime.now().strftime(format)

def criterion(name):
    if 'mse' in name:
        return mse_criterion
    elif 'sce' in name:
        return sce_criterion
    elif 'abs' in name:
        return abs_criterion
    elif 'sfce' in name:
        return sfce_criterion

def abs_criterion(logits, labels, name = 'abs_criterion'):
    return tf.reduce_mean(tf.abs(logits - labels), name = name)

def mse_criterion(logits, labels, name = 'mse_criterion'):
    return tf.reduce_mean((logits - labels)**2, name = name)

def sce_criterion(logits, labels, name = 'sce_criterion'):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), name = name)

def sfce_criterion(logits, labels, name = 'sfce_criterion'):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels), name = name)

class fakePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_count = 0
        self.fake_A= []
        self.fake_B= []
    def __call__(self, fake_A, fake_B):
        if self.maxsize == 0:
            return fake_A, fake_B
        if self.num_count < self.maxsize:
            self.fake_A.append(fake_A)
            self.fake_B.append(fake_B)
            
            self.num_count += 1
            
            return fake_A, fake_B
        
        if np.random.rand() < 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp_A = copy.copy(self.fake_A[idx])
            self.fake_A[idx] = fake_A

            tmp_B = copy.copy(self.fake_B[idx])
            self.fake_B[idx] = fake_B
            
            return tmp_A, tmp_B
        else:
            return fake_A, fake_B
        
def apply_gradient(optim, grad_and_vars, clip_lambda, is_clip):
        if is_clip:
            gradients, variables = zip(*grad_and_vars)
            gradients, _  = tf.clip_by_global_norm(gradients, clip_lambda)
            grad_and_vars = zip(gradients, variables)
            optim = optim.apply_gradients(grad_and_vars)
        else:
            optim = optim.apply_gradients(grad_and_vars)
        
        return optim, grad_and_vars
