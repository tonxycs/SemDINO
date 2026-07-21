import os
import numpy as np
import torch
from skimage import io # type: ignore
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
# from osgeo import gdal_array
import cv2

num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])

root = '/root/autodl-fs/AAAI-SemDINO/SECOND'

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time=='A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im

def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs

def read_RSimages(mode, rescale=False):
    #assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'im1')
    img_B_dir = os.path.join(root, mode, 'im2')
    label_A_dir = os.path.join(root, mode, 'label1')
    label_B_dir = os.path.join(root, mode, 'label2')
    
    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
    count = 0
    for it in data_list:
        # print(it)
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_B_path = os.path.join(label_B_dir, it)
            
            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)
            
            label_A = io.imread(label_A_path)
            label_B = io.imread(label_B_path)

            # ========== 新增：读取标签后立即转成单通道类别索引 ==========
            if len(label_A.shape) == 3:  # 如果是RGB图（3通道）
                label_A = Color2Index(label_A)  # 转成单通道 [H,W]
            if len(label_B.shape) == 3:
                label_B = Color2Index(label_B)

            labels_A.append(label_A)
            labels_B.append(label_B)
        count+=1
        if not count%500: print('%d/%d images loaded.'%(count, len(data_list)))
    
    print(labels_A[0].shape)  # 现在会输出 (512,512)，而非 (512,512,3)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')
    
    return imgs_list_A, imgs_list_B, labels_A, labels_B

class Data(data.Dataset):
    def __init__(self, mode, random_flip = False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages(mode)
    
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')
        img_B = io.imread(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')
        label_A = self.labels_A[idx]
        label_B = self.labels_B[idx]
        if self.random_flip:
            img_A, img_B, label_A, label_B = transform.rand_rot90_flip_MCD(img_A, img_B, label_A, label_B)
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label_A), torch.from_numpy(label_B)

    def __len__(self):
        return len(self.imgs_list_A)

class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'im1')
        imgB_dir = os.path.join(test_dir, 'im2')
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:]=='.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]
        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len
    

# '''
# 数据增强版本
# '''
# import os
# import numpy as np
# import torch
# from skimage import io # type: ignore
# from torch.utils import data
# import utils.transform as transform
# from torchvision.transforms import functional as F

# # ====================== 导入最强增强 ======================
# from datasets.augmentation import augmentation_compose

# num_classes = 7
# ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
# ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

# MEAN_A = np.array([113.40, 114.08, 116.45])
# STD_A  = np.array([48.30,  46.27,  48.14])
# MEAN_B = np.array([111.07, 114.04, 118.18])
# STD_B  = np.array([49.41,  47.01,  47.94])

# root = '/root/autodl-fs/AAAI-SemDINO/SECOND'

# colormap2label = np.zeros(256 ** 3)
# for i, cm in enumerate(ST_COLORMAP):
#     colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

# def Color2Index(ColorLabel):
#     data = ColorLabel.astype(np.int32)
#     idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
#     IndexMap = colormap2label[idx]
#     IndexMap = IndexMap * (IndexMap < num_classes)
#     return IndexMap

# def normalize_image(im, time='A'):
#     assert time in ['A', 'B']
#     if time=='A':
#         im = (im - MEAN_A) / STD_A
#     else:
#         im = (im - MEAN_B) / STD_B
#     return im

# def read_RSimages(mode):
#     img_A_dir = os.path.join(root, mode, 'im1')
#     img_B_dir = os.path.join(root, mode, 'im2')
#     label_A_dir = os.path.join(root, mode, 'label1')
#     label_B_dir = os.path.join(root, mode, 'label2')
    
#     data_list = os.listdir(img_A_dir)
#     imgs_list_A, imgs_list_B, labels_A, labels_B = [], [], [], []
#     count = 0
#     for it in data_list:
#         if it[-4:]=='.png':
#             img_A_path = os.path.join(img_A_dir, it)
#             img_B_path = os.path.join(img_B_dir, it)
#             label_A_path = os.path.join(label_A_dir, it)
#             label_B_path = os.path.join(label_B_dir, it)
            
#             imgs_list_A.append(img_A_path)
#             imgs_list_B.append(img_B_path)
            
#             label_A = io.imread(label_A_path)
#             label_B = io.imread(label_B_path)

#             if len(label_A.shape) == 3:
#                 label_A = Color2Index(label_A)
#             if len(label_B.shape) == 3:
#                 label_B = Color2Index(label_B)

#             labels_A.append(label_A)
#             labels_B.append(label_B)
#         count+=1
#         if not count%500:
#             print('%d/%d images loaded.'%(count, len(data_list)))
    
#     print(labels_A[0].shape)
#     print(str(len(imgs_list_A)) + ' ' + mode + ' images loaded.')
    
#     return imgs_list_A, imgs_list_B, labels_A, labels_B

# # ==============================================================================
# # ====================== 🔥 加强版 Data 类（增强已加入）======================
# # ==============================================================================
# class Data(data.Dataset):
#     def __init__(self, mode, random_flip=False):
#         self.random_flip = random_flip
#         self.imgs_list_A, self.imgs_list_B, self.labels_A, self.labels_B = read_RSimages(mode)

#     def get_mask_name(self, idx):
#         return os.path.split(self.imgs_list_A[idx])[-1]

#     def __getitem__(self, idx):
#         # 1. 读取原始图像
#         img_A = io.imread(self.imgs_list_A[idx])
#         img_B = io.imread(self.imgs_list_B[idx])
#         label_A = self.labels_A[idx]
#         label_B = self.labels_B[idx]

#         # 2. 🔥 训练模式：开启超强增强
#         if self.random_flip:
#             gt_mask = (label_A != label_B).astype(np.uint8)
#             sample = {
#                 "img1": img_A,
#                 "img2": img_B,
#                 "mask1": label_A,
#                 "mask2": label_B,
#                 "gt_mask": gt_mask
#             }
#             # 调用超强增强
#             sample = augmentation_compose(sample)
#             img_A = sample["img1"]
#             img_B = sample["img2"]
#             label_A = sample["mask1"]
#             label_B = sample["mask2"]

#         # 3. 归一化
#         img_A = normalize_image(img_A, 'A')
#         img_B = normalize_image(img_B, 'B')

#         # 4. 转张量返回
#         return (
#             F.to_tensor(img_A),
#             F.to_tensor(img_B),
#             torch.from_numpy(label_A).long(),
#             torch.from_numpy(label_B).long()
#         )

#     def __len__(self):
#         return len(self.imgs_list_A)

# # ====================== 测试集（不变）======================
# class Data_test(data.Dataset):
#     def __init__(self, test_dir):
#         self.imgs_A = []
#         self.imgs_B = []
#         self.mask_name_list = []
#         imgA_dir = os.path.join(test_dir, 'im1')
#         imgB_dir = os.path.join(test_dir, 'im2')
#         data_list = os.listdir(imgA_dir)
#         for it in data_list:
#             if it[-4:]=='.png':
#                 self.imgs_A.append(io.imread(os.path.join(imgA_dir, it)))
#                 self.imgs_B.append(io.imread(os.path.join(imgB_dir, it)))
#                 self.mask_name_list.append(it)
#         self.len = len(self.imgs_A)

#     def get_mask_name(self, idx):
#         return self.mask_name_list[idx]

#     def __getitem__(self, idx):
#         img_A = normalize_image(self.imgs_A[idx], 'A')
#         img_B = normalize_image(self.imgs_B[idx], 'B')
#         return F.to_tensor(img_A), F.to_tensor(img_B)

#     def __len__(self):
#         return self.len