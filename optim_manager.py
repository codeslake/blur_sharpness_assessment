import tensorflow as tf

class OptimManager(object):
    def __init__(self, optim='Adam', beta1 = 0.9, is_clip = False, clip_lambda = 0):
        self.is_clip = is_clip
        self.clip_lambda = clip_lambda
        self.beta1 = 0.0
        
    def apply_gradient(self, optim, grad_and_vars):
        if self.is_clip:
            gradients, variables = zip(*grad_and_vars)
            gradients, _  = tf.clip_by_global_norm(gradients, self.clip_lambda)
            grad_and_vars = zip(gradients, variables)
            optim = optim.apply_gradients(grad_and_vars)
        else:
            optim = optim.apply_gradients(grad_and_vars)
        
        return optim, grad_and_vars
    
    def get_optim_and_grad_vars(self, learning_rate, loss, var_list):
        optim = tf.train.AdamOptimizer(learning_rate, beta1 = self.beta1)
        grad_and_vars = optim.compute_gradients(loss, var_list = var_list)
        optim, grad_and_vars = self.apply_gradient(optim, grad_and_vars)
        
        return optim, grad_and_vars
