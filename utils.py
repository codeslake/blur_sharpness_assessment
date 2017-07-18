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

def get_images(image_paths, idx_range, is_crop = True, crop_mode = 'random', bboxes = None,  resize_h = 224, resize_w = None, is_grayscale = False):
    resize_w = resize_h if resize_w is None else resize_w
    
    image_paths = image_paths.flatten()
    image_paths_batch = np.take(image_paths, idx_range, axis = 0)
    bboxes_batch = np.take(bboxes, idx_range, axis = 0)
    
    images = []
    for i in np.arange(len(image_paths_batch)):
        image_path = image_paths_batch[i]
        #print '[file name]', image_path
        bbox = bboxes_batch[i]
        images.extend([get_image(image_path, is_crop, crop_mode, bbox, resize_h, resize_w, is_grayscale)])
        
    if is_grayscale:
        images = np.array(images).astype(np.float32)[:, :, :, None]
    else:
        images = np.array(images).astype(np.float32)

    return images
    
def get_image(image_path, is_crop, crop_mode, bbox, resize_h, resize_w, is_grayscale):
    return augment(imread(image_path, is_grayscale), is_crop, crop_mode, bbox, resize_h, resize_w)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imread(path, is_grayscale = False):
    if is_grayscale:
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode = 'RGB').astype(np.float)

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

def random_center_crop(x, npx, bbox, resize_h, resize_w):
    # bbox = [x-left, y-top, width, height]
    # 1. find bbox coordinates
    img_h, img_w = x.shape[:2]
    bbox_x = bbox[0]
    bbox_y = bbox[1]
    bbox_w = bbox[2]
    bbox_h = bbox[3]

    # 1-1. edit offlined bbox width or height
    if bbox_x + bbox_w > img_w:
        dist = (bbox_x + bbox_w) - img_w
        bbox_w -= dist
    if bbox_y + bbox_h > img_h:
        dist = (bbox_y + bbox_h) - img_h
        bbox_h -= dist

    # 2. find crop window which satisfy the ratio
    ratio = np.random.uniform(low = 0.75, high = 0.9)
    crop_h = crop_w = int(np.sqrt((bbox_w * bbox_h) / ratio))
    
    # 2-1. make sure it is bigger than bbox and smaller than the image
    if crop_h < bbox_w or crop_h < bbox_h:
        #print"**[Crop Box] smaller than bbox!!!"
        crop_h = crop_w = bbox_w if bbox_w > bbox_h else bbox_h

    if crop_h > npx:
        #print"**crop box bigger than image!!!"
        # augment & return
        return scipy.misc.imresize(x, [resize_h, resize_w])
        
    # 3. find the crop offset which satisfies the bounding problem
    # 3-1. find the possible crop range
    crop_xmin = max(0, bbox_x + bbox_w - crop_w)
    crop_xmax = min(bbox_x, img_w - crop_w)
    crop_x_dist = abs(crop_xmax - crop_xmin)
    crop_ymin = max(0, bbox_y + bbox_h - crop_h)
    crop_ymax = min(bbox_y, img_h - crop_h)
    crop_y_dist = abs(crop_ymax - crop_ymin)

    # 3-2. select the offset from normal distribution
    rand = np.random.normal(loc=0.5, scale=0.1)
    rand = 0.0 if rand < 0 else 1.0 if rand > 1.0 else rand
    crop_x = int(min(crop_xmin,crop_xmax) + crop_x_dist * rand)
    
    rand = np.random.normal(loc=0.5, scale=0.1)
    rand = 0.0 if rand < 0 else 1.0 if rand > 1.0 else rand
    crop_y = int(min(crop_ymin, crop_ymax) + crop_y_dist * rand)
    
    '''
    print '********************************* SUM *********************************'
    print '[img]', 'width:', img_w, ', height:', img_h
    print '[ratio]', 'expected: ', ratio, ', actual: ', (bbox_w * bbox_h)/ (crop_w * crop_h)
    print '[bbox]', 'x:', bbox_x, ', y:', bbox_y, ', w:', bbox_w, ', h:', bbox_h
    print '[crop_range]', 'xmin:', crop_xmin, ', xmax:', crop_xmax, ', ymin:', crop_ymin, ', ymax:', crop_ymax, ', xdist:', crop_x_dist, ', ydist:', crop_y_dist
    print '[crop_box]', 'x:', crop_x, ', y:', crop_y, ', w:', crop_w, ', h:', crop_h
    print '********************************* SUM *********************************'
    '''
    # 4. crop
    x = x[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    
    # 5. randomly flips the image
    rand = np.random.normal(loc=0.5, scale=0.5)
    rand = 0.0 if rand < 0 else 1.0 if rand > 1.0 else rand
    x = np.flip(x, 1) if rand > 0.5 else x
    
    return scipy.misc.imresize(x, [resize_h, resize_w])

def random_crop(x, crop_h, crop_w, resize_h, resize_w):
    rand = np.random.normal(loc=0.5, scale=0.5)
    rand = 0.0 if rand < 0 else 1.0 if rand > 1.0 else rand
    start_idx = np.round((x.shape[:2] - np.array([crop_h, crop_w])) * rand).astype(int)
    end_idx = (start_idx + np.array([crop_h, crop_w])).astype(int)

    rand = np.random.normal(loc=0.5, scale=0.5)
    rand = 0.0 if rand < 0 else 1.0 if rand > 1.0 else rand
    x = np.flip(x, 1) if rand > 0.5 else x
    
    return scipy.misc.imresize(x[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]], [resize_h, resize_w])
        
def augment(image, is_crop, crop_mode, bbox, resize_h, resize_w):
    # npx : # of pixels width/height of image
    h, w = image.shape[:2]
    npx = w if h > w else h
        
    if is_crop:
        if crop_mode is 'random':
            image = random_crop(image, npx, npx, resize_h, resize_w)
        elif crop_mode is 'center':
            image = center_crop(image, npx, npx, resize_h, resize_w)
        elif crop_mode is 'random_bbox':
            image = random_center_crop(image, npx, bbox, resize_h, resize_w)
    else:
        image = image

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
        '''
        for filename in fnmatch.filter(filenames, '/*.JPEG'):
            file_path.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '/*.PNG'):
            file_path.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '/*.jpeg'):
            file_path.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '/*.png'):
            file_path.append(os.path.join(root, filename))
        '''
        for filename in fnmatch.filter(filenames, '/*.jpg'):
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
