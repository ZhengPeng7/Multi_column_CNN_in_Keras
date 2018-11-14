import os
import cv2
import csv
import numpy as np
from scipy.io import loadmat
import get_density_map_gaussian


dataset = 'B'
dataset_name = ''.join(['shanghaitech_part_', dataset])
path = ''.join(['../data/original/ShanghaiTech/part_', dataset, '/test_data/images/'])
gt_path = ''.join(['../data/original/ShanghaiTech/part_', dataset, '/test_data/ground-truth/'])
gt_path_csv = ''.join(['../data/original/ShanghaiTech/part_', dataset, '/test_data/ground-truth_csv/'])
if not os.path.exists(gt_path_csv):
    os.makedirs(gt_path_csv)
if dataset == 'A':
    num_images = 182
else:
    num_images = 316

for i in range(1, num_images+1):    
    if i % 10 == 0:
        print('Processing {}/{} files'.format(i, num_images), '\nwriting to {}'.format(''.join([gt_path_csv, 'IMG_', str(i), '.csv'])))
    image_info = loadmat(''.join((gt_path, 'GT_IMG_', str(i), '.mat')))['image_info']
    input_img_name = ''.join((path, 'IMG_', str(i), '.jpg'))
    im = cv2.imread(input_img_name, 0)
    annPoints =  image_info[0][0][0][0][0] - 1
    im_density = get_density_map_gaussian.get_density_map_gaussian(im, annPoints)
    with open(''.join([gt_path_csv, 'IMG_', str(i), '.csv']), 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(im_density)
