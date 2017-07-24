import pandas as pd
import numpy as np
import os
import pickle
import scipy.misc
import sys

def generate_data_pickle(abs_path, filenames, scores, label):
    print "writing .."

    filenum = filenames.shape[0]
    valnum = 100
    s0 = np.arange(filenum)
    np.random.shuffle(s0)
    
    train_filenames = filenames[s0][valnum:]
    val_filenames = filenames[s0][:valnum]
    
    train_scores = scores[s0][valnum:]
    val_scores = scores[s0][:valnum]
    
    sum(train_filenames, 'train_filenames')
    sum(val_filenames, 'val_filenames')
    sum(train_scores, 'train_scores')
    sum(val_scores, 'val_scores')
    
    postfix = '_scaled'
    with open(abs_path + '/' + label + postfix + '_filenames_train.pickle', 'w') as f:
        pickle.dump(train_filenames, f)
    with open(abs_path + '/' + label + postfix + '_filenames_val.pickle', 'w') as f:
        pickle.dump(val_filenames, f)
        
    with open(abs_path + '/' + label + postfix + '_scores_train.pickle', 'w') as f:
        pickle.dump(train_scores, f)
    with open(abs_path + '/' + label + postfix + '_scores_val.pickle', 'w') as f:
        pickle.dump(val_scores, f)
        
    print "writing .. Done"

def get_data(abs_path, image_path, label_filename):
    label_path = os.path.join(abs_path, label_filename)

    df_filenames = pd.read_csv(label_path,
                            delim_whitespace = True,
                            header = None).astype(int)
    
    df_scores = pd.read_csv(label_path,
                               delim_whitespace = True,
                               header = None).astype(int)

    filenames = []
    
    scores = []
    weights = np.arange(10) + 1
    for i in xrange(0, df_filenames.shape[0]):
        filename = df_filenames.iloc[i][0].tolist()
        
        try:
            scipy.misc.imread(os.path.join(image_path, str(filename) + '.jpg'), mode = 'RGB').astype(np.float)
        except:
            print os.path.join(image_path, str(filename) + '.jpg')
            continue
            
        filenames.extend([os.path.join(image_path, str(filename) + '.jpg')])

        score = np.array(df_scores.iloc[i][2:12].tolist())
        score = (np.sum((weights * score)) / float(np.sum(score))) - 5.
        scores.extend([score])
        
    return np.array(filenames), np.array(scores)

def sum(list, name):
    print '****' + name + '****'
    print 'Total  :', len(list)
    print 'Shape  :', list.shape
    print 'Sample :', list[0], '\n'

def main():
    abs_path = '/data1/AVA/AVA_classified/255000'
    
    print abs_path

    label = 'motion_blur'
    label_filename = label + '.txt'
    image_path = '/data1/AVA/image'
    
    filenames, scores = get_data(abs_path, image_path, label_filename)
    sum(filenames, 'filenames')
    sum(scores, 'scores')

    generate_data_pickle(abs_path, filenames, scores, label)

if __name__ == "__main__":
    main()






