import pandas as pd
import numpy as np
import os
import pickle
import sys


def generate_scores_pickle(abs_path, scores):
    print "writing .."
    scores_train =
    scores_test =
    with open(abs_path + '/' + 'scores_defocus_blur.pickle', 'w') as f:
        pickle.dump(scores, f)
    print "writing .. Done"

def get_all_scores(abs_path, score):
    scores_path = os.path.join(abs_path, score)
    
    df_scores = pd.read_csv(scores_path, delim_whitespace = True, header = None)
    scores = df_scores[1].tolist()

    return np.array(scores)

def get_sepa_scores(abs_path, score):
    with open(abs_path+ '/' + score, 'rb') as f:
        scores = pickle.load(f)
        scores = np.asarray(scores)
    
    return np.array(scores)
    
def get_scores(abs_path, image_path, label_filename):
    label_path = os.path.join(abs_path, label_filename)

    df_scores = pd.read_csv(label_path,
                            delim_whitespace = True,
                            header = None).astype(int)

    scores = []
    weights = np.arange(10) + 1
    for i in xrange(0, df_scores.shape[0]):
        score = np.array(df_scores.iloc[i][2:12].tolist())
        score = np.sum((weights * score)) / np.sum(score) / 10.
        scores.extend([score])
        
    return np.array(scores)

def sum(list, name):
    print '****' + name + '****'
    print 'Total  :', len(list)
    print 'Shape  :', list.shape
    print 'Sample :', list[0], '\n'

def main():
    abs_path = '/data1/AVA/AVA_classified/255000'
    
    print abs_path

    label_filename = 'defocus_blur.txt'
    image_path = '/data1/AVA/image'
    

    scores = get_scores(abs_path, image_path, label_filename)
    sum(scores, 'scores')

    generate_scores_pickle(abs_path, scores)

if __name__ == "__main__":
    main()






