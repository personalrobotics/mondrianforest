import os
import sys
sys.path.append("/home/guohaz/rf/mondrianforest")
import numpy as np
from config import config
from process_data import load_pushing_data
import pickle

import mondrianforest
from sklearn.model_selection import cross_val_score, ShuffleSplit

if __name__ == "__main__":
    if os.path.exists(config.checkpoint_filename):
        print('Load saved checkpoint: {}'.format(config.checkpoint_filename))
        with open(config.checkpoint_filename, 'rb') as ckpt:
            forest = pickle.load(ckpt) 
    else:
        forest = mondrianforest.MondrianForestClassifier(n_tree=config.n_tree)
    X, y = load_pushing_data.load_dataset()
    print("Finished loading the data")
    cv = ShuffleSplit(n_splits=config.n_splits, test_size=config.test_size, random_state=0)
    scores = cross_val_score(forest, X, y, cv=cv)
    print(scores.mean(), scores.std())
    with open(config.checkpoint_filename, 'wb') as ckpt:
        pickle.dump(forest, ckpt)
