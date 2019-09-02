'''
@Description: 
@Version: 1.0
@Author: Taoye Yin
@Date: 2019-08-30 15:01:56
@LastEditors: Taoye Yin
@LastEditTime: 2019-09-02 15:37:40
'''
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import cv2

os.chdir(sys.path[0])

folder_list = ['I', 'II']
train_boarder = 112


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    line_parts = line.strip().split()
    img_name = 'E:/Desktop/project/face_dection' + line_parts[0][2:]
    # rect = list(map(int, list(map(float, line_parts[1:5]))))
    # landmarks = list(map(float, line_parts[5: len(line_parts)]))
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    x = list(map(float, line_parts[5::2]))
    y = list(map(float, line_parts[6::2]))
    landmarks = list(zip(x, y))
    # print (landmarks)
    return img_name, rect, landmarks


class Normalize(object):
    #有object表示继承对象 python3 里也可以不写
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_resize = np.asarray(
                            image.resize((train_boarder, train_boarder), Image.BILINEAR),
                            dtype=np.float32)       # Image.ANTIALIAS)
        image = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks
                }


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        #一个类实例要变成一个可调用对象，只需要实现一个特殊方法__call__()。
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0) # 1 * c * h *w
        image = image.transpose((0, 2, 1))
        landmarks = torch.from_numpy(landmarks)
        # print (landmark.size())
        return {'image': torch.from_numpy(image),
                'landmarks': landmarks} #将numpy 转化为tensor  ToTensor也可以


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # 返回是的tuple,本质是迭代器
        img_name, rect, landmarks = parse_line(self.lines[idx])
        # image
        # img_name = '../data/I/001424.jpg'
        with Image.open(img_name).convert('L') as img:  
            img_crop = img.crop(tuple(rect))        
            landmarks = np.array(landmarks).astype(np.float32)
            # img_crop = np.array(img_crop).astype(np.float32)
            new_landmarks = []
            for i in range(len(landmarks)):
                landmarks[i][0]-=rect[0]
                landmarks[i][0]=float((landmarks[i][0]/img_crop.size[0]) *112)
                new_landmarks.append(landmarks[i][0])
                landmarks[i][1]-=rect[1]
                landmarks[i][1]=float((landmarks[i][1]/img_crop.size[1]) *112)
                new_landmarks.append(landmarks[i][1])
		# you should let your landmarks fit to the train_boarder(112)
		# please complete your code under this blank
		# your code:
            new_landmarks = np.array(new_landmarks)
        # print (type(img_crop))
		
        sample = {'image': img_crop, 'landmarks': new_landmarks}
        sample = self.transform(sample)
        return sample


def load_data(phase):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),                # do channel normalization
            ToTensor()]                 # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, transform=tsfm)
    return data_set


def get_train_test_set():
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set, valid_set


if __name__ == '__main__':
    train_set = load_data('train')
    for i in range(1, len(train_set)):
        sample = train_set[i]
        img = sample['image']
        landmarks = sample['landmarks']
		## 请画出人脸crop以及对应的landmarks
		# please complete your code under this blank
        # print (type(img))
        
        img = np.array(img.permute(2,1,0))
        # print (img.shape)
        # print (landmarks.size())
        # print (landmarks)
        # plt.imshow(img)
        x = list(map(int, landmarks[0::2]))
        y = list(map(int, landmarks[1::2]))
        landmarks = list(zip(x, y))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for p in landmarks:
            # print (p)
            cv2.circle(img,p,3,(0,0,255),-1)

        cv2.imshow('face_dection',img)
		
		
		
		

        key = cv2.waitKey()
        if key == 27:
            exit(0)
    cv2.destroyAllWindows()
