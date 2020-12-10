import glob
import json

import cv2
import h5py
import numpy as np
import tqdm

IMAGE_SIZE = (512, 256)
SKELETON_SIZE = 20
HOME = 'C:/Users/Alberto Ursino/Desktop/IntellIj Local Files/Convert-BADJA-json/'


def replace_slash(path):
    return path.replace('\\', '/')


def images_sampler(url_list):
    frames = []
    img_idx = 0
    dict_img = {}
    for url in url_list:
        for image_file in tqdm.tqdm(glob.glob(url)):
            dict_img[replace_slash(image_file)] = img_idx
            img_idx += 1
            img = cv2.imread(image_file)
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append(img)
    return frames, dict_img, img_idx


def delete_useless_joints(xy_list):
    new_xy_list = []
    for i in range(8, 11):
        new_xy_list.append(xy_list[i])
    for i in range(12, 16):
        new_xy_list.append(xy_list[i])
    for i in range(18, 21):
        new_xy_list.append(xy_list[i])
    for i in range(22, 26):
        new_xy_list.append(xy_list[i])
    for i in range(28, 29):
        new_xy_list.append(xy_list[i])
    for i in range(31, 34):
        new_xy_list.append(xy_list[i])
    for i in range(35, 37):
        new_xy_list.append(xy_list[i])
    return new_xy_list


def swap_coordinates(xy_list):
    for i in range(len(xy_list)):
        x = xy_list[i][0]
        xy_list[i][0] = xy_list[i][1]
        xy_list[i][1] = x
    return xy_list


def resize_coordinates(xy_list, rx, ry):
    resized_xy_list = []
    for xy in xy_list:
        resized_xy_list.append([xy[0] * rx, xy[1] * ry])
    return resized_xy_list


def insert_nan(joints_list, skeleton_size, index):
    for idx in range(skeleton_size):
        if joints_list[index][idx][0] + joints_list[index][idx][1] == 0:
            joints_list[index][idx] = [np.nan, np.nan]
    return joints_list


def fill_annotations_ds(original_res, badja_json_paths, index_dict, hf_file):
    for json_p, res in zip(badja_json_paths, original_res):
        resize_ratio_x = IMAGE_SIZE[0] / res[0]
        resize_ratio_y = IMAGE_SIZE[1] / res[1]
        joints_coordinates = []
        num_ann = 0
        j_file = open(json_p)
        j_data = json.load(j_file)
        for xy in j_data:
            joints = delete_useless_joints(xy['joints'])
            joints = swap_coordinates(joints)
            joints = resize_coordinates(joints, resize_ratio_x, resize_ratio_y)
            joints_coordinates.append(joints)
            image_name = xy['segmentation_path']
            image_index = index_dict.get(HOME + image_name)
            joints_coordinates = insert_nan(joints_coordinates, SKELETON_SIZE, num_ann)
            hf_file["annotations"][image_index] = joints_coordinates[num_ann]
            hf_file["annotated"][image_index] = np.ones(SKELETON_SIZE, int)
            num_ann += 1
        j_file.close()


if __name__ == '__main__':

    hf = h5py.File(HOME + 'datasets/annotation_set_{}_{}.h5'.format(IMAGE_SIZE[0], IMAGE_SIZE[1]), 'w')

    # Creating 'skeleton' dataset

    skeleton_hf = h5py.File(HOME + 'skeleton.h5', 'r')
    hf.create_dataset('skeleton', data=skeleton_hf['skeleton'])

    # Creating 'images' dataset

    urls = [HOME + 'extra_videos/rs_dog/segmentations/*.png', HOME + 'DAVIS/Annotations/Full-Resolution/dog/*.png',
            HOME + 'DAVIS/Annotations/Full-Resolution/dog-agility/*.png']

    sampled_frames, idx_dict, num_images = images_sampler(urls)

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

    fill_annotations_ds([(1280, 720), (1920, 1080), (1920, 1080)],
                        [HOME + 'rs_dog.json', HOME + 'dog.json', HOME + 'dog_agility.json'], idx_dict, hf)

    print('annotations dataset created')
    print('annotated dataset created')

    hf.close()
