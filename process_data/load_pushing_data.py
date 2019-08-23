import os
import sys
sys.path.append("/home/guohaz/rf/mondrianforest")
import numpy as np
from config import config
import pickle

def load_dataset(obj_dir=config.obj_dir, action_dir=config.action_dir, test=False):
    action_files = sorted(os.listdir(action_dir))
    X, y = [], []
    for action_file in action_files:
        if action_file[-4:] == '.npy':
            X_file, y_file = process_single_trial(action_file[:4], obj_dir, action_dir, test)
            X += X_file
            y += y_file
    return X, y

def process_single_trial(trial_id, obj_dir, action_dir, test):
    X, y = [], []
    kernel_size = int(config.img_window_size / 2)
    action = np.load(os.path.join(action_dir, trial_id+'_action.npy'))
    before_action = np.load(os.path.join(obj_dir, trial_id+'_before.npy'))
    if not test:
        after_action = np.load(os.path.join(obj_dir, trial_id+'_after.npy'))
    for i in range(before_action.shape[0]):
        for j in range(before_action.shape[1]):
            if action[i,j] != 0:
                x_lo, x_hi = i - kernel_size, i + kernel_size + 1
                y_lo, y_hi = j - kernel_size, j + kernel_size + 1
                # X_i = []
                # X_i.append(before_action[x_lo : x_hi, y_lo : y_hi])
                # X_i.append(action[x_lo : x_hi, y_lo : y_hi])
                X_i = np.append(before_action[x_lo : x_hi, y_lo : y_hi], action[x_lo : x_hi, y_lo : y_hi])
                if np.array(X_i).shape[0] == 50:
                    X.append(X_i)
                    if not test:
                        y.append(after_action[i, j])
    return X, y