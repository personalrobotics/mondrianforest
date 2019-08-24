import os
import sys
sys.path.append("/home/guohaz/rf/mondrianforest")
import numpy as np
from config import config
from process_data import preprocess, load_pushing_data
import pickle
import cv2


def load_img():
    obj_img_path = input("object img: ")
    distance = input("Pushing distance in cm: ") # later change to /joint_state post
    angle = input("Pushing angle: ")

    obj_img = cv2.imread(obj_img_path)
    dist = {"start_point":{"transformation":[0,0]},
            "end_point":{"transformation":[distance, 0]}}

    # Get down_sized dim
    width = int(obj_img.shape[1] / config.obj_downsize_length)
    height = int(obj_img.shape[0] / config.obj_downsize_length)
    dim = (width, height)
    
    # Create image
    action_img = preprocess.create_action_img(dim, dist, angle)
    before_action_img = preprocess.create_obj_img(obj_img, dim, angle, distance)[:,:,0]
    
    # Store/return image
    np.save(os.path.join(config.test_dir, 'actions/test_action.npy'), action_img)
    np.save(os.path.join(config.test_dir, 'objects/test_before.npy'), before_action_img)
    print("Finished processing img")
    return obj_img, action_img


def predict(action_img, forest, X):
    after_action_img = np.zeros(action_img.shape)

    # For train on entire image 
    # id = 0
    
    # for i in range(0, action_img.shape[0]):
    #     for j in range(0, action_img.shape[1]):
    #         if forest.predict_proba(X[id])[1] > 0.5:
    #             after_action_img[i,j] = 1
    #         id += 1

    # For only train on action, and predict on action
    id = 0
    for i in range(action_img.shape[0]):
        for j in range(action_img.shape[1]):
            if action_img[i,j] != 0:
                if forest.predict_proba(X[id])[1] > 0.5:
                    after_action_img[i,j] = 1
                id += 1
    return after_action_img


def generate_estimation(obj_img, action_img, after_action_img):
    x_lo, x_hi, y_lo, y_hi = (230, 270, 390, 450)
    obj_color = np.average(obj_img[x_lo: x_hi, y_lo : y_hi], axis=0)
    for i in range(action_img.shape[0]):
        for j in range(action_img.shape[1]):
            if action_img[i,j] == 1: # The pixel need to be updated
                if after_action_img[i,j] == 1: # has mashed potato after pushing
                    obj_img[i * config.obj_downsize_length : (i+1) * onfig.obj_downsize_length, \
                            j * config.obj_downsize_length :  (j+1) * onfig.obj_downsize_length] = obj_color[:]
                else: # no mashed potato after pushing, fill with dish color
                    obj_img[i * config.obj_downsize_length : (i+1) * onfig.obj_downsize_length, \
                            j * config.obj_downsize_length :  (j+1) * onfig.obj_downsize_length] = config.blue_range[1]
    return obj_img

if __name__ == "__main__":
    if os.path.exists(config.checkpoint_filename):
        print('Load saved checkpoint: {}'.format(config.checkpoint_filename))
        forest = pickle.load(config.checkpoint_filename)
    else:
        print('No model exist!')
        os._exit(1)

    # Load image
    obj_img, action_img = load_img()
    
    # Create predict
    X, y = load_pushing_data.load_dataset(obj_dir=os.path.join(config.test_dir, 'objects'), \
                                          action_dir=os.path.join(config.test_dir, 'actions'), \
                                          test=True)
    after_action_img = predict(action_img, forest, X)

    # Generate and save estimate image
    estimated_img = generate_estimation(obj_img, action_img, after_action_img)
    cv2.imwrite(os.path.join(config.test_dir, 'test_estimated.png'), estimated_img)