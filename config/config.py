# General configuration

import os

'''
General
'''
img_window_size = 5 # in pixel after down sample
data_dir = "/home/guohaz/hard_data/Data/foods/pushing_data"
blue_range = ([98,0,0], [250,255,255])
cm2pixel = 410 / 18.25 # disk diameter in pixel / that in cm

'''
Object
'''
img_dir = os.path.join(data_dir, "images")
obj_dir = os.path.join(data_dir, "objects")
depth_dir = os.path.join(data_dir, "depths")
distance_filename = os.path.join(data_dir, "bags/distance.json")
obj_downsize_length = 20 # in unit of half width of the forque
obj_depth_thres = 2

'''
Action
'''
action_dir = os.path.join(data_dir, "actions")
action_width = 2 # in pixel after down sample
action_start_pos = {0: (14,20), 90: (14,20)} # angle: (h, w)

'''
Model
'''
n_tree = 30
n_splits=5
test_size=0.2
checkpoint_filename = os.path.join(data_dir, 'checkpoint.pkl')
test_dir = os.path.join(data_dir, "result")
