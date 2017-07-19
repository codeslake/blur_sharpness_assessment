from bird_manager import BirdManager
import numpy as np

birdManager = BirdManager();

filenames, captions = birdManager.get_captions('/data1/BIRD/captions/train')

traindata = np.append(filenames[:, None], captions, axis = 1)

print filenames.shape
print captions.shape




