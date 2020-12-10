import glob
import json

import cv2
import numpy as np
import h5py
import tqdm
import pandas as pd

IMAGE_SIZE = (512, 256)
SKELETON_SIZE = 20
IMG_IDX = 0
HOME = 'C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/Convert-BADJA-json/'

idx_dict = {}

hf = h5py.File(HOME + 'datasets/annotation_set_{}_{}.h5'.format(IMAGE_SIZE[0], IMAGE_SIZE[1]), 'w')
skeleton_hf = h5py.File(HOME + 'skeleton.h5', 'r')


def replace_slash(path):
    return path.replace('\\', '/')


# Creating 'skeleton' dataset

hf.create_dataset('skeleton', data=skeleton_hf['skeleton'])

# Creating 'images' dataset

sampled_frames = []
num_images = 0

for image_file in tqdm.tqdm(
        glob.glob(HOME + 'extra_videos/rs_dog/segmentations/*.png')):
    idx_dict[replace_slash(image_file)] = IMG_IDX
    IMG_IDX += 1
    num_images += 1
    img = cv2.imread(image_file)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sampled_frames.append(img)

for image_file in tqdm.tqdm(
        glob.glob(HOME + 'DAVIS/Annotations/Full-Resolution/dog/*.png')):
    idx_dict[replace_slash(image_file)] = IMG_IDX
    IMG_IDX += 1
    num_images += 1
    img = cv2.imread(image_file)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sampled_frames.append(img)

for image_file in tqdm.tqdm(
        glob.glob(HOME + 'DAVIS/Annotations/Full-Resolution/dog-agility/*.png')):
    idx_dict[replace_slash(image_file)] = IMG_IDX
    IMG_IDX += 1
    num_images += 1
    img = cv2.imread(image_file)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sampled_frames.append(img)

sampled_frames = np.concatenate(sampled_frames)
sampled_frames = np.reshape(sampled_frames, (num_images, IMAGE_SIZE[1], IMAGE_SIZE[0], 1))

hf.create_dataset('images', data=sampled_frames)
print('images dataset created')

# Creating 'annotations' and 'annotated' datasets

none_matrix = np.zeros((num_images, SKELETON_SIZE, 2))
for i in range(num_images):
    for j in range(SKELETON_SIZE):
        none_matrix[i][j] = np.nan

hf.create_dataset('annotations', dtype=np.float64, data=none_matrix)
hf.create_dataset('annotated', shape=(num_images, SKELETON_SIZE), dtype=bool)


def swap_coordinates(xy_list):
    swapped_xy_list = xy_list
    for i in range(len(xy_list)):
        x = swapped_xy_list[i][0]
        swapped_xy_list[i][0] = swapped_xy_list[i][1]
        swapped_xy_list[i][1] = x
    return swapped_xy_list


def resize_coordinates(xy_list, rx, ry):
    resized_xy_list = []
    for xy in xy_list:
        resized_xy_list.append([xy[0] * rx, xy[1] * ry])
    return resized_xy_list


def delete_useless_joints(xy_list):
    pruned_xy_list = []
    for i in range(8, 11):
        pruned_xy_list.append(xy_list[i])
    for i in range(12, 16):
        pruned_xy_list.append(xy_list[i])
    for i in range(18, 21):
        pruned_xy_list.append(xy_list[i])
    for i in range(22, 26):
        pruned_xy_list.append(xy_list[i])
    for i in range(28, 29):
        pruned_xy_list.append(xy_list[i])
    for i in range(31, 34):
        pruned_xy_list.append(xy_list[i])
    for i in range(35, 37):
        pruned_xy_list.append(xy_list[i])
    return pruned_xy_list


def insert_nan(joints_list, skeleton_size, index):
    for idx in range(skeleton_size):
        if joints_list[index][idx][0] + joints_list[index][idx][1] == 0:
            joints_list[index][idx] = [np.nan, np.nan]
    return joints_list


resize_ratio_x = IMAGE_SIZE[0] / 1280
resize_ratio_y = IMAGE_SIZE[1] / 720
joints_coordinates = []
num_ann = 0
j_file = open(HOME + 'rs_dog.json')
j_data = json.load(j_file)
for xy in j_data:
    joints = delete_useless_joints(xy['joints'])
    joints = swap_coordinates(joints)
    joints = resize_coordinates(joints, resize_ratio_x, resize_ratio_y)
    joints_coordinates.append(joints)
    image_name = xy['segmentation_path']
    image_index = idx_dict.get(HOME + image_name)
    #joints_coordinates = insert_nan(joints_coordinates, SKELETON_SIZE, num_ann)
    hf["annotations"][image_index] = joints_coordinates[num_ann]
    hf["annotated"][image_index] = np.ones(SKELETON_SIZE, int)
    num_ann += 1

resize_ratio_x = IMAGE_SIZE[0] / 1920
resize_ratio_y = IMAGE_SIZE[1] / 1080
joints_coordinates = []
num_ann = 0
j_file = open(HOME + 'dog.json')
j_data = json.load(j_file)
for xy in j_data:
    joints = delete_useless_joints(xy['joints'])
    joints = swap_coordinates(joints)
    joints = resize_coordinates(joints, resize_ratio_x, resize_ratio_y)
    joints_coordinates.append(joints)
    image_name = xy['segmentation_path']
    image_index = idx_dict.get(HOME + image_name)
    #joints_coordinates = insert_nan(joints_coordinates, SKELETON_SIZE, num_ann)
    hf["annotations"][image_index] = joints_coordinates[num_ann]
    hf["annotated"][image_index] = np.ones(SKELETON_SIZE, int)
    num_ann += 1

resize_ratio_x = IMAGE_SIZE[0] / 1920
resize_ratio_y = IMAGE_SIZE[1] / 1080
joints_coordinates = []
num_ann = 0
j_file = open(HOME + 'dog_agility.json')
j_data = json.load(j_file)
for xy in j_data:
    joints = delete_useless_joints(xy['joints'])
    joints = swap_coordinates(joints)
    joints = resize_coordinates(joints, resize_ratio_x, resize_ratio_y)
    joints_coordinates.append(joints)
    image_name = xy['segmentation_path']
    image_index = idx_dict.get(HOME + image_name)
    joints_coordinates = insert_nan(joints_coordinates, SKELETON_SIZE, num_ann)
    hf["annotations"][image_index] = joints_coordinates[num_ann]
    hf["annotated"][image_index] = np.ones(SKELETON_SIZE, int)
    num_ann += 1

print('annotations dataset created')
print('annotated dataset created')

hf.close()
j_file.close()
