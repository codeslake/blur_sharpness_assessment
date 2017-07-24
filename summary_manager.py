import tensorflow as tf

class SummaryManager(object):
    def __init__(self):
        self.sum_mark = '_sum'
        self.sum_vars = None
    
    def add_image_sum(self, var, name):
        tf.identity(var, name = 'image_' + name + '/' + self.sum_mark)
    
    def add_histogram_sum(self, var, name):
        tf.identity(var, name = 'hist_' + name + '/' + self.sum_mark)
        
    def add_text_sum(self, var, name):
        tf.identity(var, name = 'text_' + name + '/' + self.sum_mark)
    
    def get_sum_marked_name(self, name):
        return name + '/' + self.sum_mark
    
    def set_sum_vars(self, graph):
        ops = [var for var in tf.contrib.graph_editor.get_tensors(graph) if self.sum_mark in var.name]
        self.sum_vars = [graph.get_tensor_by_name(var.name) for var in ops]
    
    def get_merged_summary(self, net_vars = None, grad_norm = None, name = None):
        if net_vars is None:
            return tf.summary.merge([
                [tf.summary.image(var.name.split('/')[-2].replace(':', '_'), var) for var in self.sum_vars if 'image' in var.name],
                [tf.summary.text(var.name.split('/')[-2].replace(':', '_'), var) for var in self.sum_vars if 'text' in var.name],
                [tf.summary.histogram('intermediate_variable_histrogram/'+ var.name.split('/')[-2].replace(':', '_'), var) for var in self.sum_vars if 'hist' in var.name]])
        else:
            if grad_norm is not None:
                return tf.summary.merge([
                    [tf.summary.histogram(name + '/' + '/'.join(var.name.split('/')[2:]).replace(':', '_'), var) for var in net_vars],
                    [tf.summary.scalar(name + '/grads_norm', grad_norm)],
                    [tf.summary.scalar(name + '/' + var.name.split('/')[-2].replace(':', '_'), var) for var in self.sum_vars if name in var.name]])
            else:
                return tf.summary.merge([
                    [tf.summary.histogram(name + '/' + '/'.join(var.name.split('/')[2:]).replace(':', '_'), var) for var in net_vars],
                    [tf.summary.scalar(name + '/' + var.name.split('/')[-2].replace(':', '_'), var) for var in self.sum_vars if name in var.name]])
                
            
        
