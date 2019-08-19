
# coding: utf-8

import os
import sys
sys.path.append("/Users/kyleghz/GitHub-Workspace/mondrianforest")
import numpy as np
import cv2
import json
import math
import matplotlib.pyplot as plt
from config import config


def create_obj_img(obj_img, dim, angle, distance):
    '''
    Downsample the image so that the length for each block is half width of the 
    forque. Then, convert the image to binary where mashed_potato is 1 and 0 otherwise.
    The pixel out of the path is also set to 0
    
    obj_img: The original object image.
    dim:     The down_sized dimension 
    angle:   The pushing angle
    distance:The pushing distance after down sized
    '''
    # Downsize
    resized = cv2.resize(obj_img, dim, interpolation = cv2.INTER_CUBIC)
    
    # Remove blue
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lower_blue = np.array(config.blue_range[0])
    upper_blue = np.array(config.blue_range[1])

    mask = cv2.inRange(hsv, lower_blue, upper_blue) -255
    take = np.where(mask==0, mask, -mask)

    res = cv2.bitwise_and(resized,resized, mask=take)
    
    # Convert to binary
    res = 1.0 * (res  > 0)
    
    # set out-of-range pixel to 0
    start_position = config.action_start_pos[angle]
    boundary = int(config.img_window_size / 2) + 1
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            if angle == 0:
                if y < start_position[1] - boundary                  or y > start_position[1] + boundary                  or x > start_position[0]                  or x < start_position[0] - distance:
                    res[x,y] = 0
            else: # angle = 90
                if x < start_position[0] - boundary                  or x > start_position[0] + boundary                  or y > start_position[1]                  or y < start_position[1] - distance:
                    res[x,y] = 0
    return res


def create_action_img(dim, dist, angle):
    '''
    Create a binary image where action is from 0 to 1 representing the distance pushed
    and 0 otherwise.
    
    dim:     The down_sized dimension 
    dist:    The pushing distance information stored in json file
    angle:   The pushing angle
    '''
    action_img = np.zeros((dim[1], dim[0]))
    start_in_tf = np.array(dist["start_point"]["transformation"])
    end_in_tf = np.array(dist["end_point"]["transformation"])
    distance = np.sqrt(np.sum((end_in_tf[0:2] - start_in_tf[0:2]) ** 2)) # unit in m

    # convert distance to downsized length 
    distance_in_pixel = math.ceil(distance * config.cm2pixel * 100 / config.obj_downsize_length)
    start_x, start_y = config.action_start_pos[angle]
    for i in range(distance_in_pixel):
        dist_value = 1.0 * (i+1) / distance_in_pixel
        if angle == 0:
            action_img[start_x - i, start_y] = dist_value
            action_img[start_x - i, start_y + 1] = dist_value
        else: # angle = 90
            action_img[start_x, start_y - i] = dist_value
            action_img[start_x + 1, start_y - i] = dist_value
    return action_img, distance_in_pixel


def load_distance_file():
    '''
    Load corresponding distance json file in config file
    '''
    with open(config.distance_filename, 'r') as distFile:
        dist_info = json.load(distFile)
    return dist_info


def convert_imgs(trial_id, dist, angle):
    '''
    Convert images to down_sized, binary images
    
    trial_id:The id for current trial
    dist:    The pushing distance information stored in json file
    angle:   The pushing angle
    '''
    trial_i = "{0:0=4d}".format(trial_id)
    start_img = cv2.imread(os.path.join(config.img_dir, trial_i + "_1_start.png"))
    finish_img = cv2.imread(os.path.join(config.img_dir, trial_i+ "_5_finish.png"))
    
    # Get down_sized dim
    width = int(start_img.shape[1] / config.obj_downsize_length)
    height = int(start_img.shape[0] / config.obj_downsize_length)
    dim = (width, height)
    
    # Create image
    action_img, distance = create_action_img(dim, dist, angle)
    before_action_img = create_obj_img(start_img, dim, angle, distance)
    after_action_img = create_obj_img(finish_img, dim, angle, distance)
    
    # Store image
#     cv2.imwrite(os.path.join(config.action_dir, trial_i+'_action.png'), action_img)
#     cv2.imwrite(os.path.join(config.obj_dir, trial_i+'_before.png'), before_action_img)
#     cv2.imwrite(os.path.join(config.obj_dir, trial_i+'_after.png'), after_action_img)
    np.save(os.path.join(config.action_dir, trial_i+'_action.npy'), action_img)
    np.save(os.path.join(config.obj_dir, trial_i+'_before.npy'), before_action_img)
    np.save(os.path.join(config.obj_dir, trial_i+'_after.npy'), after_action_img)
    print("Finished {}".format(trial_id))


def main():
    if not os.path.isdir(config.obj_dir):
        os.makedirs(config.obj_dir)
    if not os.path.isdir(config.action_dir):
        os.makedirs(config.action_dir)
    dist_info = load_distance_file()
    for trial_i in dist_info:
        trial_id = int(trial_i)
        angle = 0
        # hard_coded angle information
        if trial_id > 99 and trial_id < 200          or trial_id > 299 and trial_id < 400:
            angle = 90
        convert_imgs(trial_id, dist_info[trial_i], angle)


def show_img(filename):
    img = np.load(filename)
    
    # print(img)
    plt.imshow(img)
    plt.title('img')
    plt.show()


main()


# img = cv2.imread(os.path.join(config.img_dir, "0014_5_finish.png"))
# depth = cv2.imread(os.path.join(config.depth_dir, "0014_5_finish.png"), cv2.IMREAD_ANYDEPTH)

# result = create_obj_img(img, 0)
# plt.imshow(result)
# plt.title('Downsize first')
# plt.show()

