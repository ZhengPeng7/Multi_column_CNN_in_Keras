from keras.models import load_model, model_from_json
import tensorflow as tf
import numpy as np
import sys
import os 
import cv2
import math
from utils import generate_generator, gen_imgPaths_and_labelPaths, monitor_mae, monitor_mse
cv2.filter2D

if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('usage: python3 test.py A(or B)')
    exit()
print('dataset:', dataset)

img_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/images/'
den_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/ground_truth_csv/'

def data_pre_test():
    print('loading test data from dataset', dataset, '...')
    img_names = os.listdir(img_path)
    img_names = img_names
    img_num = len(img_names)

    data = []
    for i in range(1, img_num + 1)[:3]:
        if i % 50 == 0:
            print(i, '/', img_num)
        name = 'IMG_' + str(i) + '.jpg'
        #print(name + '****************************')
        img = cv2.imread(img_path + name, 0)
        img = np.array(img)
        img = (img - 127.5) / 128
        #print(img.shape)
        den = np.loadtxt(den_path + name[:-4] + '.csv', delimiter = ",")
        #print(den.shape)
        den_sum = np.sum(den)
        data.append([img, den_sum])
            
    print('load data finished.')
    return data
    
data = data_pre_test()

# model = load_model('./weights/mcnn_B.hdf5', custom_objects={'monitor_mae': monitor_mae, 'monitor_mse': monitor_mse})
model = model_from_json(open('../projects_cloned/crowd-counting-MCNN/model.json').read())
model.load_weights('../projects_cloned/crowd-counting-MCNN/weights.h5')

mae = 0
mse = 0
for d in data[:3]:
    inputs = np.reshape(d[0], [1, 768, 1024, 1])
    outputs = np.squeeze(model.predict(inputs))
    np.savetxt('./outputs.txt', outputs)
    import matplotlib.pyplot as plt
    fg, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
    ax0.imshow(np.squeeze(inputs))
    ax1.imshow(outputs, cmap='gray')
    plt.show()
    den = d[1]
    c_act = np.sum(den)
    c_pre = np.sum(outputs)
    print('pre:', c_pre, 'act:', c_act)
    mae += abs(c_pre - c_act)
    mse += (c_pre - c_act) * (c_pre - c_act)
mae /= len(data)
mse /= len(data)
mse = math.sqrt(mse)

print('#############################')
print('mae:', mae, 'mse:', mse)
