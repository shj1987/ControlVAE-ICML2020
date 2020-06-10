"""
Fun: divide the data into training and testing data
"""

import glob
import shutil
import os,random


def _split_data(file_path, test_path):
    '''split the data into train and test data'''
    image_list = glob.glob(os.path.join(file_path, "*.jpg"))
    filenames = random.sample(image_list, 10000)
    for img in filenames:
        shutil.move(img, test_path)


def main():
    file_path = './data/CelebA/img_align_celeba/'
    test_path = './data/test/'
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    ## randomly choose the images to test data
    _split_data(file_path, test_path)


if __name__ == "__main__":
    main()
    