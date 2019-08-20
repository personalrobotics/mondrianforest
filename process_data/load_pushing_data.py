import os
import sys
sys.path.append("/Users/kyleghz/GitHub-Workspace/mondrianforest")
import numpy as np
from config import config

import mondrianforest
from sklearn.model_selection import cross_val_score, ShuffleSplit

def load_dataset():
    action_files = sorted(os.listdir(config.action_dir))
    X, y = [], []
    for action_file in action_files:
        if action_file[-4:] == '.npy':
            X_file, y_file = process_single_trial(action_file[:4])
            X += X_file
            y += y_file
    return X, y

def process_single_trial(trial_id):
    X, y = [], []
    kernel_size = int(config.img_window_size / 2)
    action = np.load(os.path.join(config.action_dir, trial_id+'_action.npy'))
    before_action = np.load(os.path.join(config.obj_dir, trial_id+'_before.npy'))
    after_action = np.load(os.path.join(config.obj_dir, trial_id+'_after.npy'))
    for i in range(before_action.shape[0]):
        for j in range(before_action.shape[1]):
            if before_action[i,j] == 1:
                x_lo, x_hi = i - kernel_size, i + kernel_size + 1
                y_lo, y_hi = j - kernel_size, j + kernel_size + 1
                # X_i = []
                # X_i.append(before_action[x_lo : x_hi, y_lo : y_hi])
                # X_i.append(action[x_lo : x_hi, y_lo : y_hi])
                X_i = np.append(before_action[x_lo : x_hi, y_lo : y_hi], action[x_lo : x_hi, y_lo : y_hi])
                if np.array(X_i).shape[0] != 0:
                    X.append(X_i)
                    # print(trial_id, i, j)
                    y.append(after_action[i, j])
    return X, y

if __name__ == "__main__":
    X, y = load_dataset()
    forest = mondrianforest.MondrianForestClassifier(n_tree=10)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    scores = cross_val_score(forest, X, y, cv=cv) 
    print(scores.mean(), scores.std())