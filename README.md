# Multi_column_CNN_in_Keras_for_crowd_counting
A simple and unofficial Keras version implementation of Multi-column CNN for crowd counting.



Multi-column CNN is the crowd counting algorithm proposed in a CVPR 2016 paper ["Single Image Crowd Counting via Multi Column Convolutional Neural Network"](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf).



## Data preprocessing:

The data can be downloaded on [dropbox](<https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0>) or [Baidu Disk](<http://pan.baidu.com/s/1nuAYslz>) can't be used directly without some preprocessing.

1. Create directory `data/original` in the root path of this repository, then move the decompressed `ShanghaiTech` to it.

2. Run the `create_gt_test_set_shtech.py` to generate the csv files for test which can be loaded as:

   ![csv_sample](images/csv_sample.jpg)

3. Run the `create_training_set_shtech.py` to generate selected images and csv files randomly for training and validation. in `formatted_trainval`.

> These three python files in `data_preparation` are adapted from the original MATLAB version preprocessing implemented in [this mcnn repository](https://github.com/svishwa/crowdcount-mcnn#data-setup) in pytorch.



## Results:

> Some outputs after 200 epochs -- loss curves:![loss](images/loss.jpg)

+ Good one:

![result_sample_good](images/result_sample_good.jpg)

+ Acceptable one:

![result_sample_normal](images/result_sample_normal.jpg)

+ Bad one:

![result_sample_bad](images/result_sample_bad.jpg)