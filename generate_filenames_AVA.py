import pandas as pd
import numpy as np
import os
import pickle
import sys


def generate_filenames_pickle(abs_path, filenames):
    print "writing .."
    with open(abs_path + '/' + 'filenames.pickle', 'w') as f:
        pickle.dump(filenames, f)
    print "writing .. Done"

def get_all_filenames(abs_path, filename):
    filenames_path = os.path.join(abs_path, filename)
    
    df_filenames = pd.read_csv(filenames_path, delim_whitespace = True, header = None)
    filenames = df_filenames[1].tolist()

    return np.array(filenames)

def get_sepa_filenames(abs_path, filename):
    with open(abs_path+ '/' + filename, 'rb') as f:
        filenames = pickle.load(f)
        filenames = np.asarray(filenames)
    
    return np.array(filenames)
    
def get_filenames(abs_path, image_path, label_filename):
    label_path = os.path.join(abs_path, label_filename)

    df_filenames = pd.read_csv(label_path,
                            delim_whitespace = True,
                            header = None).astype(int)

    filenames = []
    for i in xrange(0, df_filenames.shape[0]):
        filename = df_filenames.iloc[i][0].tolist()
        filenames.extend([os.path.join(image_path, str(filename) + '.jpg')])
        
    return np.array(filenames)

def sum(list, name):
    print '****' + name + '****'
    print 'Total  :', len(list)
    print 'Shape  :', list.shape
    print 'Sample :', list[0], '\n'

def main(is_train):
    if is_train:
        abs_path = '/data1/AVA/AVA_dataset'
    else :
        abs_path = '/data1/AVA/AVA_dataset'
    
    print abs_path

    label_filename = 'AVA.txt'
    image_path = '/data1/AVA/image'
    

    filenames = get_filenames(abs_path, image_path, label_filename)
    sum(filenames, 'filenames')

    generate_filenames_pickle(abs_path, filenames)

if __name__ == "__main__":
    arg = True if 'True' in sys.argv[1] else False if 'False' in sys.argv[1] else None
    main(arg)






