import pickle
import numpy as np
import os

class AVAManager(object):
    def __init__(self, abs_path, type, train=False):
    
        self.abs_path = os.path.join(abs_path, 'AVA_classified', '255000')
        self.file_num = None
        
        if train:
            self.image_file_name = type + '_filenames_train.pickle'
            self.score_file_name = type + '_scores_train.pickle'
        else:
            self.image_file_name = type + '_filenames_val.pickle'
            self.score_file_name = type + '_scores_val.pickle'
            
        self.derange_index = None # derange idx
        
        self.data = {}
        self.init()
        
    def get_data(self):
        # Shuffle
        s0 = np.arange(self.file_num)
        np.random.shuffle(s0)
        
        return self.data['filenames'][s0], self.data['scores'][s0]

    def init(self):
        # load file_names
        with open(os.path.join(self.abs_path, self.image_file_name), 'rb') as f:
            filenames = np.array(pickle.load(f)) #[file_num]
            
        # load scores
        with open(os.path.join(self.abs_path, self.score_file_name), 'rb') as f:
            scores = np.array(pickle.load(f)) #[file_num]
            
        # Filenames & Scores
        self.file_num = filenames.shape[0]
        
        #set original data
        self.data['filenames'] = filenames
        self.data['scores'] = scores
