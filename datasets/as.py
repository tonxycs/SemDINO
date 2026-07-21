# import os
# import numpy as np
# from skimage import io
# from tqdm import tqdm

# root = '~/HRSCD/train/A'
# root_b = '~/HRSCD/train/B'

# def compute_mean_std(folder_path):
#     pixel_num = 0
#     channel_sum = np.zeros(3)
#     channel_sum_sq = np.zeros(3)

#     for name in tqdm(os.listdir(folder_path)):
#         img = io.imread(os.path.join(folder_path, name))
#         img = np.array(img, dtype=np.float32)

#         H, W, C = img.shape
#         pixel_num += H * W

#         channel_sum += np.sum(img, axis=(0, 1))
#         channel_sum_sq += np.sum(img ** 2, axis=(0, 1))

#     mean = channel_sum / pixel_num
#     std = np.sqrt(channel_sum_sq / pixel_num - mean ** 2)
#     return mean, std


# mean_A, std_A = compute_mean_std(root)


# mean_B, std_B = compute_mean_std(root_b)

# print("MEAN_A = np.array([%.2f, %.2f, %.2f])" % tuple(mean_A))
# print("STD_A  = np.array([%.2f, %.2f, %.2f])" % tuple(std_A))
# print("MEAN_B = np.array([%.2f, %.2f, %.2f])" % tuple(mean_B))
# print("STD_B  = np.array([%.2f, %.2f, %.2f])" % tuple(std_B))


# import os
# import random
# import shutil


# root = "~/HRSCD"
# train_src = os.path.join(root, "train")
# val_dst = os.path.join(root, "val")
# sub_dirs = ["A", "B", "labelA", "labelB"]
# random.seed(42)  


# def create_folder(base):
#     if not os.path.exists(base):
#         os.makedirs(base)
#     for sub in sub_dirs:
#         p = os.path.join(base, sub)
#         if not os.path.exists(p):
#             os.makedirs(p)

# create_folder(val_dst)


# img_A_path = os.path.join(train_src, "A")
# all_names = [f for f in os.listdir(img_A_path) if f.endswith(".png")]
# random.shuffle(all_names)
# total_train_samples = len(all_names)



# # test = 0.2T
# # train_all = 0.8T

# test_A = os.path.join(root, "test", "A")
# test_num = len([f for f in os.listdir(test_A) if f.endswith(".png")])
# val_num = test_num  


# val_names = all_names[:val_num]
# new_train_names = all_names[val_num:]


# def move_batch(name_list):
#     for name in name_list:
#         for sub in sub_dirs:
#             src = os.path.join(train_src, sub, name)
#             dst = os.path.join(val_dst, sub, name)
#             shutil.move(src, dst)

# move_batch(val_names)


# def write_txt(save_path, name_list):
#     with open(save_path, "w", encoding="utf-8") as f:
#         for n in name_list:
#             f.write(n + "\n")

# write_txt(os.path.join(root, "train_list.txt"), new_train_names)
# write_txt(os.path.join(root, "val_list.txt"), val_names)

# test_names = [f for f in os.listdir(test_A) if f.endswith(".png")]
# write_txt(os.path.join(root, "test_list.txt"), test_names)


# print(f"{len(new_train_names)}")
# print("train_list.txt / val_list.txt / test_list.txt")

from skimage import io


colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


lab = io.imread("/root/autodl-fs/HRSCD/train/labelA/14-2012-0420-6895-LA93-0M50-E080_r0002_c0028.png")
lab = Color2Index(lab)
print(np.unique(lab))
